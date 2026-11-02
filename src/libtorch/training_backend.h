#ifndef RLBG_LIBTORCH_TRAINING_BACKEND_H
#define RLBG_LIBTORCH_TRAINING_BACKEND_H

#include <cstdint>

#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <torch/torch.h>

#include <training/basic_training_loop.h>
#include <training/training_backend.h>

#include <libtorch/learner.h>
#include <libtorch/model_inference.h>
#include <libtorch/transformer_muzero.h>

namespace rlbg::libtorch {

namespace s = std;
namespace t = torch;

struct TrainingConfig : BasicTrainingLoopConfig {
  TransformerMuZeroConfig model;
  LearnerConfig           learner;
  s::string               device = "auto";
  // latents kept on the GPU per inference copy, 0 for twice what the
  // searches in flight can hold, -1 to transfer latents to the host instead
  int64_t                 latent_pool = 0;
};

// latents the self-play and reanalyze searches of a training run can hold at
// once, doubled for actors running at different speeds
inline int64_t default_latent_pool(const TrainingConfig& c){
  int threads = c.num_actors > 0 ? c.num_actors : (int)s::max(1u, s::thread::hardware_concurrency());
  int64_t searches = (int64_t)threads * c.selfplay.games_per_actor;
  int64_t per_search = c.selfplay.num_simulations + 1;
  return 2 * (searches * per_search + c.server.max_batch);
}

// the training loop's view of LibTorch: a Learner on the GPU, ModelInference
// copies in inference_dtype and .pt snapshots (save_model files)
struct TrainingBackend : rlbg::TrainingBackend {
  explicit TrainingBackend(const TrainingConfig& config)
  : mModel(config.model), mLearner(config.learner), mDevice(parse_device(config.device)),
    mDtype(parse_dtype(config.inference_dtype)),
    mLatentPool(config.latent_pool < 0 ? 0 : config.latent_pool > 0 ? config.latent_pool : default_latent_pool(config)),
    mEvalLatentPool(config.latent_pool < 0 ? 0
                    : 2 * ((int64_t)config.evaluator.games * (config.evaluator.simulations + 1) + config.server.max_batch))
  {
    // half precision on the CPU is slow or unsupported
    if (not mDevice.is_cuda()) mDtype = t::kFloat;
  }

  struct TorchLearner : TrainingLearner {
    TorchLearner(const TransformerMuZeroConfig& model, const LearnerConfig& config, t::Device device,
                 t::ScalarType inference_dtype)
    : mLearner(model, config, device), mInferenceDtype(inference_dtype), mDevice(device) {}

    LearnerStats train_step(const TrainingBatch& batch) override { return mLearner.train_step(batch); }
    int64_t step() const override { return mLearner.step(); }
    void save_checkpoint(const s::string& dir) override { mLearner.save_checkpoint(dir); }
    bool load_checkpoint(const s::string& dir) override { return mLearner.load_checkpoint(dir); }
    void save_weights(const s::string& file) override { mLearner.save_weights(file); }
    s::function<void()> weights_update(SnapshotInference& target) override {
      auto* inference = dynamic_cast<ModelInference*>(&target);
      if (not inference) throw s::invalid_argument("LibTorch weights need a LibTorch inference");
      auto copy = s::make_shared<TransformerMuZero>(mLearner.copy_model(mInferenceDtype));
      return [inference, copy]{ inference->set_weights(*copy); };
    }
    s::string describe() const override {
      TransformerMuZero& m = const_cast<Learner&>(mLearner).model();
      return s::format("model TransformerMuZero ({}) with {} parameters on {}", m->config.describe(),
                       m->num_parameters(), mDevice.str());
    }

    Learner       mLearner;
    t::ScalarType mInferenceDtype;
    t::Device     mDevice;
  };

  void seed(uint64_t seed) override { t::manual_seed(seed); }
  s::unique_ptr<TrainingLearner> make_learner() override {
    return s::make_unique<TorchLearner>(mModel, mLearner, mDevice, mDtype);
  }
  s::shared_ptr<SnapshotInference> make_inference() override {
    return s::make_shared<ModelInference>(mModel, mDevice, mDtype, mLatentPool);
  }
  s::shared_ptr<SnapshotInference> make_evaluation_inference() override {
    return s::make_shared<ModelInference>(mModel, mDevice, mDtype, mEvalLatentPool);
  }
  int latent_size() const override {
    return mLatentPool > 0 ? ModelInference::LATENT_HANDLE_SIZE : (int)(mModel.num_tokens * mModel.dims);
  }
  int64_t latent_pool() const { return mLatentPool; }
  s::string weights_extension() const override { return ".pt"; }

protected:
  TransformerMuZeroConfig mModel;
  LearnerConfig           mLearner;
  t::Device               mDevice;
  t::ScalarType           mDtype;
  int64_t                 mLatentPool;
  int64_t                 mEvalLatentPool;
};

}  // namespace rlbg::libtorch

#endif//RLBG_LIBTORCH_TRAINING_BACKEND_H
