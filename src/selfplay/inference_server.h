#ifndef RLBG_SELFPLAY_INFERENCE_SERVER_H
#define RLBG_SELFPLAY_INFERENCE_SERVER_H

#include <cstdint>
#include <memory>

#include <mlx_util/nn.h>

#include <model/model_inference.h>
#include <selfplay/batching_inference_server.h>

namespace rlbg {

namespace s = std;

// the batching inference server over an MLX model, whose weights are
// replaced with set_weights
struct InferenceServer : BatchingInferenceServer {
  InferenceServer(s::shared_ptr<UpdatableInference> backend, const InferenceServerConfig& config)
  : BatchingInferenceServer(backend, config), mUpdatable(s::move(backend))
  {}

  // installed between batches; version tags the trajectories it produces
  void set_weights(nn::Params weights, uint32_t version){
    post_update([backend = mUpdatable, w = s::move(weights)]{ backend->set_weights(w); }, version);
  }

protected:
  s::shared_ptr<UpdatableInference> mUpdatable;
};

}  // namespace rlbg

#endif//RLBG_SELFPLAY_INFERENCE_SERVER_H
