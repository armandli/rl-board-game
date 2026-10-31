#ifndef RLBG_TRAINING_LEARNER_H
#define RLBG_TRAINING_LEARNER_H

#include <cstdint>

#include <string>

#include <mlx_util/nn.h>

#include <model/muzero_model.h>
#include <training/batch.h>

namespace rlbg {

namespace s = std;

struct LearnerStats {
  int64_t step = 0;
  double  loss = 0.;
  double  policy = 0.;
  double  value = 0.;
  double  reward = 0.;
  double  consistency = 0.;
  double  reconstruction = 0.;
  double  ownership = 0.;
  double  grad_norm = 0.;
  double  learning_rate = 0.;
};

// running mean of learner statistics between progress reports; the
// learning rate is the latest
struct LearnerStatsAverage {
  void add(const LearnerStats& st){
    mSum.loss += st.loss;
    mSum.policy += st.policy;
    mSum.value += st.value;
    mSum.reward += st.reward;
    mSum.consistency += st.consistency;
    mSum.reconstruction += st.reconstruction;
    mSum.ownership += st.ownership;
    mSum.grad_norm += st.grad_norm;
    mSum.learning_rate = st.learning_rate;
    mSum.step = st.step;
    ++mCount;
  }
  int count() const { return mCount; }
  // the mean so far, then starts over
  LearnerStats take(){
    LearnerStats r = mSum;
    if (mCount > 0){
      r.loss /= mCount;
      r.policy /= mCount;
      r.value /= mCount;
      r.reward /= mCount;
      r.consistency /= mCount;
      r.reconstruction /= mCount;
      r.ownership /= mCount;
      r.grad_norm /= mCount;
    }
    mSum = LearnerStats();
    mCount = 0;
    return r;
  }

protected:
  LearnerStats mSum;
  int          mCount = 0;
};

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
