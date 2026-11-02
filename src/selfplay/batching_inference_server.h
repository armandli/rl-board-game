#ifndef RLBG_SELFPLAY_BATCHING_INFERENCE_SERVER_H
#define RLBG_SELFPLAY_BATCHING_INFERENCE_SERVER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <search/inference.h>

namespace rlbg {

namespace s = std;

struct InferenceServerConfig {
  // most positions evaluated in one network call
  int    max_batch = 2048;
  // a batch is launched once this fraction of the active clients have
  // submitted, or after max_wait; below 1 lets clients split into groups
  // that alternate between searching on the CPU and waiting on the GPU
  double client_fraction = 0.5;
  int    max_wait_us = 2000;
};

// batches the requests of many self-play threads into large network calls on
// a dedicated thread, which keeps the GPU busy while the CPU threads search.
// Clients call infer() and block until their results are ready. Independent
// of the ML framework: the backend is any Inference, and new weights arrive
// as an update function the server runs between two batches.
struct BatchingInferenceServer : Inference {
  BatchingInferenceServer(s::shared_ptr<Inference> backend, const InferenceServerConfig& config)
  : mBackend(s::move(backend)), mConfig(config)
  {}

  void infer(const InferenceRequests& requests, InferenceResults& results) override {
    Pending p{&requests, &results};
    s::unique_lock<s::mutex> lock(mMutex);
    if (mStop){
      // shutting down: answer with zeros so callers can unwind
      lock.unlock();
      results.num_initial = requests.num_initial();
      results.latent_size = latent_size();
      results.num_actions = num_actions();
      results.latents.assign((size_t)requests.size() * latent_size(), latent_t(0));
      results.policy_logits.assign((size_t)requests.size() * num_actions(), 0.f);
      results.values.assign(requests.size(), 0.f);
      results.rewards.assign(requests.size(), 0.f);
      return;
    }
    mQueue.push_back(&p);
    mQueuedPositions += requests.size();
    mServerCv.notify_one();
    mClientCv.wait(lock, [&]{ return p.done; });
  }
  int observation_size() const override { return mBackend->observation_size(); }
  int latent_size() const override { return mBackend->latent_size(); }
  int num_actions() const override { return mBackend->num_actions(); }

  // serves requests until stop() is called
  void run(){
    s::vector<Pending*> batch;
    while (true){
      {
        s::unique_lock<s::mutex> lock(mMutex);
        mServerCv.wait(lock, [&]{ return mStop or not mQueue.empty(); });
        if (mStop and mQueue.empty()) break;
        auto deadline = s::chrono::steady_clock::now() + s::chrono::microseconds(mConfig.max_wait_us);
        auto enough = [&]{
          int want = s::max(1, (int)s::ceil(mClients * mConfig.client_fraction));
          return mStop or mQueuedPositions >= mConfig.max_batch or (int)mQueue.size() >= want;
        };
        mServerCv.wait_until(lock, deadline, enough);
        batch.clear();
        int positions = 0;
        size_t taken = 0;
        while (taken < mQueue.size() and (taken == 0 or positions + mQueue[taken]->requests->size() <= mConfig.max_batch)){
          positions += mQueue[taken]->requests->size();
          batch.push_back(mQueue[taken]);
          ++taken;
        }
        mQueue.erase(mQueue.begin(), mQueue.begin() + taken);
        mQueuedPositions -= positions;
      }
      apply_pending_update();
      serve(batch);
      {
        s::lock_guard<s::mutex> lock(mMutex);
        for (Pending* p : batch) p->done = true;
      }
      mClientCv.notify_all();
    }
  }

  void stop(){
    {
      s::lock_guard<s::mutex> lock(mMutex);
      mStop = true;
    }
    mServerCv.notify_all();
    mClientCv.notify_all();
  }

  // clients announce themselves so the server knows how many to wait for
  void add_client(){
    s::lock_guard<s::mutex> lock(mMutex);
    ++mClients;
  }
  void remove_client(){
    {
      s::lock_guard<s::mutex> lock(mMutex);
      --mClients;
    }
    mServerCv.notify_one();
  }

  // update runs on the server thread before the next batch, replacing any
  // update not yet run; version tags the trajectories played after it
  void post_update(s::function<void()> update, uint32_t version){
    s::lock_guard<s::mutex> lock(mUpdateMutex);
    mPendingUpdate = s::move(update);
    mPendingVersion = version;
  }
  uint32_t weights_version() const { return mVersion.load(); }

  uint64_t num_batches() const { return mBatches.load(); }
  uint64_t num_positions() const { return mPositions.load(); }

protected:
  struct Pending {
    const InferenceRequests* requests;
    InferenceResults*        results;
    bool                     done = false;
  };

  void apply_pending_update(){
    s::function<void()> update;
    uint32_t version = 0;
    {
      s::lock_guard<s::mutex> lock(mUpdateMutex);
      if (not mPendingUpdate) return;
      update = s::move(mPendingUpdate);
      mPendingUpdate = nullptr;
      version = mPendingVersion;
    }
    update();
    mVersion = version;
  }

  void serve(s::vector<Pending*>& batch){
    int obs = observation_size();
    int lat = latent_size();
    int na = num_actions();
    mMerged.observation_size = obs;
    mMerged.latent_size = lat;
    mMerged.clear();
    int total_initial = 0;
    for (Pending* p : batch){
      const InferenceRequests& r = *p->requests;
      mMerged.features.insert(mMerged.features.end(), r.features.begin(), r.features.end());
      total_initial += r.num_initial();
    }
    for (Pending* p : batch){
      const InferenceRequests& r = *p->requests;
      mMerged.latents.insert(mMerged.latents.end(), r.latents.begin(), r.latents.end());
      mMerged.actions.insert(mMerged.actions.end(), r.actions.begin(), r.actions.end());
    }
    mBackend->infer(mMerged, mMergedResults);
    mBatches += 1;
    mPositions += mMerged.size();

    // scatter: a client's initial results start at its initial offset, its
    // recurrent results at total_initial plus its recurrent offset
    int io = 0;
    int ro = total_initial;
    auto copy = [&](InferenceResults& out, int dst, int src){
      s::memcpy(out.latents.data() + (size_t)dst * lat, mMergedResults.latent(src), lat * sizeof(latent_t));
      s::memcpy(out.policy_logits.data() + (size_t)dst * na, mMergedResults.logits(src), na * sizeof(float));
      out.values[dst] = mMergedResults.values[src];
      out.rewards[dst] = mMergedResults.rewards[src];
    };
    for (Pending* p : batch){
      const InferenceRequests& r = *p->requests;
      InferenceResults& out = *p->results;
      int ni = r.num_initial();
      int nr = r.num_recurrent();
      out.num_initial = ni;
      out.latent_size = lat;
      out.num_actions = na;
      out.latents.resize((size_t)(ni + nr) * lat);
      out.policy_logits.resize((size_t)(ni + nr) * na);
      out.values.resize(ni + nr);
      out.rewards.resize(ni + nr);
      for (int i = 0; i < ni; ++i) copy(out, i, io + i);
      for (int i = 0; i < nr; ++i) copy(out, ni + i, ro + i);
      io += ni;
      ro += nr;
    }
  }

  s::shared_ptr<Inference>  mBackend;
  InferenceServerConfig     mConfig;

  s::mutex                  mMutex;
  s::condition_variable     mServerCv;
  s::condition_variable     mClientCv;
  s::vector<Pending*>       mQueue;
  int                       mQueuedPositions = 0;
  int                       mClients = 0;
  bool                      mStop = false;

  s::mutex                  mUpdateMutex;
  s::function<void()>       mPendingUpdate;
  uint32_t                  mPendingVersion = 0;
  s::atomic<uint32_t>       mVersion{0};

  s::atomic<uint64_t>       mBatches{0};
  s::atomic<uint64_t>       mPositions{0};

  InferenceRequests         mMerged;
  InferenceResults          mMergedResults;
};

}  // namespace rlbg

#endif//RLBG_SELFPLAY_BATCHING_INFERENCE_SERVER_H
