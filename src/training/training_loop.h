#ifndef RLBG_TRAINING_TRAINING_LOOP_H
#define RLBG_TRAINING_TRAINING_LOOP_H

#include <atomic>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include <muzero_game.h>

#include <encoder/encoder.h>
#include <model/model_inference.h>
#include <model/muzero_model.h>
#include <training/basic_training_loop.h>
#include <training/muzero_learner.h>
#include <training/training_backend.h>

namespace rlbg {

namespace s = std;
namespace mx = mlx::core;

struct TrainingLoopConfig : BasicTrainingLoopConfig {
  MuZeroLearnerConfig learner;
};

// the training loop's view of MLX: MuZeroLearner, ModelInference copies and
// .safetensors snapshots
struct MlxTrainingBackend : TrainingBackend {
  using ModelFactory = s::function<s::shared_ptr<MuZeroModel>()>;

  MlxTrainingBackend(ModelFactory make_model, const MuZeroLearnerConfig& learner, int observation_size,
                     const s::string& inference_dtype)
  : mMakeModel(s::move(make_model)), mLearner(learner), mObservationSize(observation_size),
    mDtype(parse_dtype(inference_dtype)), mLatentSize(mMakeModel()->latent_size())
  {}

  struct Learner : TrainingLearner {
    Learner(s::shared_ptr<MuZeroModel> model, const MuZeroLearnerConfig& config)
    : mModel(model), mLearner(model, config) {}

    LearnerStats train_step(const TrainingBatch& batch) override { return mLearner.train_step(batch); }
    int64_t step() const override { return mLearner.step(); }
    void save_checkpoint(const s::string& dir) override { mLearner.save_checkpoint(dir); }
    bool load_checkpoint(const s::string& dir) override { return mLearner.load_checkpoint(dir); }
    void save_weights(const s::string& file) override { mModel->save_weights(file); }
    s::function<void()> weights_update(SnapshotInference& target) override {
      auto* updatable = dynamic_cast<UpdatableInference*>(&target);
      if (not updatable) throw s::invalid_argument("MLX weights need an MLX inference");
      return [updatable, w = mLearner.weights()]{ updatable->set_weights(w); };
    }
    s::string describe() const override {
      return s::format("model {} with {} parameters", mModel->type_name(), mModel->num_parameters());
    }

    s::shared_ptr<MuZeroModel> mModel;
    MuZeroLearner              mLearner;
  };

  void seed(uint64_t seed) override { mx::random::seed(seed); }
  s::unique_ptr<TrainingLearner> make_learner() override {
    return s::make_unique<Learner>(mMakeModel(), mLearner);
  }
  s::shared_ptr<SnapshotInference> make_inference() override {
    return s::make_shared<ModelInference>(mMakeModel(), mObservationSize, mDtype);
  }
  int latent_size() const override { return mLatentSize; }
  s::string weights_extension() const override { return ".safetensors"; }

protected:
  ModelFactory        mMakeModel;
  MuZeroLearnerConfig mLearner;
  int                 mObservationSize;
  mx::Dtype           mDtype;
  int                 mLatentSize;
};

// MuZero training of an MLX model; see BasicTrainingLoop
template <MuZeroGame Game>
struct TrainingLoop {
  using ModelFactory = MlxTrainingBackend::ModelFactory;

  TrainingLoop(const TrainingLoopConfig& config, const Encoder<Game>* encoder, ModelFactory make_model)
  : mLoop(config, encoder, s::make_shared<MlxTrainingBackend>(
      s::move(make_model), config.learner, encoder->observation_size(), config.inference_dtype))
  {}

  void run(const s::atomic<bool>& interrupt){ mLoop.run(interrupt); }

protected:
  BasicTrainingLoop<Game> mLoop;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_TRAINING_LOOP_H
