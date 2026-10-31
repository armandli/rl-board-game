#ifndef RLBG_TRAINING_SAMPLE_RATIO_H
#define RLBG_TRAINING_SAMPLE_RATIO_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace rlbg {

namespace s = std;

// keeps training and self-play at a fixed ratio of sample_reuse training
// samples per self-play position, whatever the machine's balance of CPU and
// GPU: the learner waits when it gets ahead, self-play waits when it gets more
// than selfplay_lead positions ahead. Nothing trains before min_positions
// positions exist. The lead is at least one batch worth of positions; with
// that, the two sides never wait on each other at the same time.
struct SampleRatio {
  SampleRatio(int batch_size, double sample_reuse, size_t min_positions, size_t selfplay_lead)
  : mBatchSize(batch_size), mReuse(sample_reuse), mMinPositions(min_positions),
    mLead(s::max((double)selfplay_lead, s::ceil(batch_size / sample_reuse)))
  {}

  // whether the learner may take step number `step` + 1, given the moves
  // played so far and the positions available for sampling
  bool learner_may_step(int64_t step, double played, size_t replay_positions) const {
    if (replay_positions < mMinPositions) return false;
    return (double)(step + 1) * mBatchSize <= mReuse * played;
  }

  // whether self-play should wait for the learner, which has taken `step`
  // steps. Never before replay holds min_positions finished positions: moves
  // of games still in progress are not sampled yet, and a learner waiting
  // for data must not wait on self-play too.
  bool selfplay_should_pause(int64_t step, double played, size_t replay_positions) const {
    if (replay_positions < mMinPositions) return false;
    double wanted = (double)step * mBatchSize / mReuse;
    return played > s::max(wanted, (double)mMinPositions) + mLead;
  }

protected:
  int    mBatchSize;
  double mReuse;
  size_t mMinPositions;
  double mLead;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_SAMPLE_RATIO_H
