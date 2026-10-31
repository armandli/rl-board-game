#ifndef RLBG_SELFPLAY_INFERENCE_SERVER_H
#define RLBG_SELFPLAY_INFERENCE_SERVER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <mlx_util/nn.h>

#include <model/model_inference.h>
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
// Clients call infer() and block until their results are ready.
struct InferenceServer : Inference {
  InferenceServer(s::shared_ptr<UpdatableInference> backend, const InferenceServerConfig& config);

  void infer(const InferenceRequests& requests, InferenceResults& results) override;
  int observation_size() const override { return mBackend->observation_size(); }
  int latent_size() const override { return mBackend->latent_size(); }
  int num_actions() const override { return mBackend->num_actions(); }

  // serves requests until stop() is called
  void run();
  void stop();

  // clients announce themselves so the server knows how many to wait for
  void add_client();
  void remove_client();

  // installed between batches; version tags the trajectories it produces
  void set_weights(nn::Params weights, uint32_t version);
  uint32_t weights_version() const { return mVersion.load(); }

  uint64_t num_batches() const { return mBatches.load(); }
  uint64_t num_positions() const { return mPositions.load(); }

protected:
  struct Pending {
    const InferenceRequests* requests;
    InferenceResults*        results;
    bool                     done = false;
  };

  void serve(s::vector<Pending*>& batch);

  s::shared_ptr<UpdatableInference> mBackend;
  InferenceServerConfig             mConfig;

  s::mutex                          mMutex;
  s::condition_variable             mServerCv;
  s::condition_variable             mClientCv;
  s::vector<Pending*>               mQueue;
  int                               mQueuedPositions = 0;
  int                               mClients = 0;
  bool                              mStop = false;

  s::mutex                          mWeightsMutex;
  s::optional<nn::Params>           mPendingWeights;
  uint32_t                          mPendingVersion = 0;
  s::atomic<uint32_t>               mVersion{0};

  s::atomic<uint64_t>               mBatches{0};
  s::atomic<uint64_t>               mPositions{0};

  InferenceRequests                 mMerged;
  InferenceResults                  mMergedResults;
};

}  // namespace rlbg

#endif//RLBG_SELFPLAY_INFERENCE_SERVER_H
