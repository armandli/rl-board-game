#ifndef RLBG_TRAINING_LEARNER_STATS_H
#define RLBG_TRAINING_LEARNER_STATS_H

#include <cstdint>

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

}  // namespace rlbg

#endif//RLBG_TRAINING_LEARNER_STATS_H
