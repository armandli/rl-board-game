#ifndef RLBG_MODEL_MODEL_INFERENCE_H
#define RLBG_MODEL_MODEL_INFERENCE_H

#include <memory>
#include <mutex>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include <model/muzero_model.h>
#include <search/inference.h>

namespace rlbg {

namespace s = std;
namespace mx = mlx::core;

// "float32", "float16" or "bfloat16"
mx::Dtype parse_dtype(const s::string& name);

// an Inference whose network weights can be replaced, e.g. by the learner's
struct UpdatableInference : Inference {
  virtual void set_weights(const nn::Params& weights) = 0;
};

// runs a MuZeroModel on the calling thread; serialized by a mutex. dtype is
// the precision of the forward pass (float16 is about twice as fast as
// float32 for inference on Apple GPUs).
struct ModelInference : UpdatableInference {
  ModelInference(s::shared_ptr<MuZeroModel> model, int observation_size, mx::Dtype dtype = mx::float32);

  void infer(const InferenceRequests& requests, InferenceResults& results) override;
  int observation_size() const override { return mObservationSize; }
  int latent_size() const override { return mModel->latent_size(); }
  int num_actions() const override { return mModel->num_actions(); }

  void set_weights(const nn::Params& weights) override;
  MuZeroModel& model() { return *mModel; }

protected:
  s::shared_ptr<MuZeroModel> mModel;
  int                        mObservationSize;
  mx::Dtype                  mDtype;
  s::mutex                   mMutex;
};

}  // namespace rlbg

#endif//RLBG_MODEL_MODEL_INFERENCE_H
