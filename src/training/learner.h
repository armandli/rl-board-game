#ifndef RLBG_TRAINING_LEARNER_H
#define RLBG_TRAINING_LEARNER_H

#include <cstdint>

#include <string>

#include <mlx_util/nn.h>

#include <model/muzero_model.h>
#include <training/batch.h>
#include <training/learner_stats.h>

namespace rlbg {

namespace s = std;

// owns the trained model and its optimizer and takes gradient steps on
// training batches
struct Learner {
  virtual ~Learner() = default;

  virtual LearnerStats train_step(const TrainingBatch& batch) = 0;
  virtual int64_t step() const = 0;
  // current weights, evaluated; safe to hand to other threads
  virtual nn::Params weights() const = 0;
  virtual MuZeroModel& model() = 0;

  virtual void save_checkpoint(const s::string& dir) const = 0;
  // returns false when dir holds no checkpoint
  virtual bool load_checkpoint(const s::string& dir) = 0;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_LEARNER_H
