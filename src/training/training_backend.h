#ifndef RLBG_TRAINING_TRAINING_BACKEND_H
#define RLBG_TRAINING_TRAINING_BACKEND_H

#include <cstdint>

#include <functional>
#include <memory>
#include <string>

#include <search/inference.h>
#include <training/batch.h>
#include <training/learner_stats.h>

namespace rlbg {

namespace s = std;

// the learner as the training loop sees it: owns the trained model and its
// optimizer, takes gradient steps and persists itself. Every call comes from
// the learner thread.
struct TrainingLearner {
  virtual ~TrainingLearner() = default;

  virtual LearnerStats train_step(const TrainingBatch& batch) = 0;
  virtual int64_t step() const = 0;

  virtual void save_checkpoint(const s::string& dir) = 0;
  // returns false when dir holds no checkpoint
  virtual bool load_checkpoint(const s::string& dir) = 0;
  // current weights as a snapshot file that SnapshotInference::load_weights reads
  virtual void save_weights(const s::string& file) = 0;
  // captures the current weights and returns a function that installs them
  // into target, an inference made by the same backend; the function may run
  // on another thread while training goes on
  virtual s::function<void()> weights_update(SnapshotInference& target) = 0;

  virtual s::string describe() const = 0;
};

// what the training loop needs from an ML framework (MLX, LibTorch): the
// model's learner, inference copies for self-play and evaluation, and the
// file format of snapshots
struct TrainingBackend {
  virtual ~TrainingBackend() = default;

  virtual void seed(uint64_t seed) = 0;
  virtual s::unique_ptr<TrainingLearner> make_learner() = 0;
  // for self-play (several threads through a BatchingInferenceServer) and
  // evaluation; weights are installed with weights_update or load_weights
  virtual s::shared_ptr<SnapshotInference> make_inference() = 0;
  // for the evaluator, which runs fewer searches at once than self-play
  virtual s::shared_ptr<SnapshotInference> make_evaluation_inference(){ return make_inference(); }
  virtual int latent_size() const = 0;
  // extension of snapshot files, e.g. ".safetensors"
  virtual s::string weights_extension() const = 0;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_TRAINING_BACKEND_H
