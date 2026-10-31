#include <selfplay/inference_server.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rlbg {

InferenceServer::InferenceServer(s::shared_ptr<UpdatableInference> backend, const InferenceServerConfig& config)
  : mBackend(s::move(backend)), mConfig(config)
{}

void InferenceServer::add_client() {
  s::lock_guard<s::mutex> lock(mMutex);
  ++mClients;
}

void InferenceServer::remove_client() {
  {
    s::lock_guard<s::mutex> lock(mMutex);
    --mClients;
  }
  mServerCv.notify_one();
}

void InferenceServer::set_weights(nn::Params weights, uint32_t version) {
  s::lock_guard<s::mutex> lock(mWeightsMutex);
  mPendingWeights = s::move(weights);
  mPendingVersion = version;
}

void InferenceServer::stop() {
  {
    s::lock_guard<s::mutex> lock(mMutex);
    mStop = true;
  }
  mServerCv.notify_all();
  mClientCv.notify_all();
}

void InferenceServer::infer(const InferenceRequests& requests, InferenceResults& results) {
  Pending p{&requests, &results};
  s::unique_lock<s::mutex> lock(mMutex);
  if (mStop) {
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

void InferenceServer::run() {
  s::vector<Pending*> batch;
  while (true) {
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
      while (taken < mQueue.size() and (taken == 0 or positions + mQueue[taken]->requests->size() <= mConfig.max_batch)) {
        positions += mQueue[taken]->requests->size();
        batch.push_back(mQueue[taken]);
        ++taken;
      }
      mQueue.erase(mQueue.begin(), mQueue.begin() + taken);
      mQueuedPositions -= positions;
    }
    {
      s::lock_guard<s::mutex> lock(mWeightsMutex);
      if (mPendingWeights) {
        mBackend->set_weights(*mPendingWeights);
        mVersion = mPendingVersion;
        mPendingWeights.reset();
      }
    }
    serve(batch);
    {
      s::lock_guard<s::mutex> lock(mMutex);
      for (Pending* p : batch) p->done = true;
    }
    mClientCv.notify_all();
  }
}

void InferenceServer::serve(s::vector<Pending*>& batch) {
  int obs = observation_size();
  int lat = latent_size();
  int na = num_actions();
  mMerged.observation_size = obs;
  mMerged.latent_size = lat;
  mMerged.clear();
  int total_initial = 0;
  for (Pending* p : batch) {
    const InferenceRequests& r = *p->requests;
    mMerged.features.insert(mMerged.features.end(), r.features.begin(), r.features.end());
    total_initial += r.num_initial();
  }
  for (Pending* p : batch) {
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
  auto copy = [&](InferenceResults& out, int dst, int src) {
    s::memcpy(out.latents.data() + (size_t)dst * lat, mMergedResults.latent(src), lat * sizeof(latent_t));
    s::memcpy(out.policy_logits.data() + (size_t)dst * na, mMergedResults.logits(src), na * sizeof(float));
    out.values[dst] = mMergedResults.values[src];
    out.rewards[dst] = mMergedResults.rewards[src];
  };
  for (Pending* p : batch) {
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

}  // namespace rlbg
