#ifndef RLBG_TRAINING_BATCH_H
#define RLBG_TRAINING_BATCH_H

#include <cstdint>

#include <algorithm>
#include <mutex>
#include <random>
#include <vector>

#include <encoder/encoder.h>
#include <experience/replay_buffer.h>
#include <experience/trajectory.h>

namespace rlbg {

namespace s = std;

// a training batch of B positions unrolled K steps; step k of a sample is
// state t+k of its trajectory. Masks are 1 where the target exists:
//   policy/value/ownership: t+k is not terminal (policy also needs a full search)
//   reward: the transition t+k -> t+k+1 happened
//   state: t+k exists (reconstruction; consistency for k >= 1)
// Value targets are distributions over loss/draw/win; a final result is one
// hot, a bootstrapped value v is two hot on the support {-1, 0, +1} with
// expectation v.
struct TrainingBatch {
  int B = 0, K = 0, T = 0, F = 0, A = 0;

  s::vector<uint8_t> observations;        // B, T, F            state t
  s::vector<uint8_t> target_observations; // B, K, T, F         states t+1..t+K
  s::vector<int32_t> actions;             // B, K
  s::vector<float>   policies;            // B, K+1, A
  s::vector<float>   policy_mask;         // B, K+1
  s::vector<float>   values;              // B, K+1, 3
  s::vector<float>   value_mask;          // B, K+1
  s::vector<int32_t> rewards;             // B, K    outcome class
  s::vector<float>   reward_mask;         // B, K
  s::vector<int32_t> tokens;              // B, K+1, T  token classes of state t+k
  s::vector<int32_t> ownership;           // B, K+1, T  final token classes seen from t+k
  s::vector<float>   state_mask;          // B, K+1

  void resize(int b, int k, int t, int f, int a){
    B = b; K = k; T = t; F = f; A = a;
    observations.assign((size_t)b * t * f, 0);
    target_observations.assign((size_t)b * k * t * f, 0);
    actions.assign((size_t)b * k, 0);
    policies.assign((size_t)b * (k + 1) * a, 0.f);
    policy_mask.assign((size_t)b * (k + 1), 0.f);
    values.assign((size_t)b * (k + 1) * 3, 0.f);
    value_mask.assign((size_t)b * (k + 1), 0.f);
    rewards.assign((size_t)b * k, 1);
    reward_mask.assign((size_t)b * k, 0.f);
    tokens.assign((size_t)b * (k + 1) * t, 0);
    ownership.assign((size_t)b * (k + 1) * t, 0);
    state_mask.assign((size_t)b * (k + 1), 0.f);
  }
};

// EfficientZero's off-policy correction of value targets (Ye et al. 2021).
// A game played by an older network has a value target that bootstraps from
// a recent search value instead of the final result: state j's target is the
// rewards of the next l moves plus the root value of state j+l, where
//   l = max(min_horizon, horizon - age / horizon_decay_steps)
// shrinks with the game's age in learner steps. The root value must have been
// searched at most max_value_age steps ago (by self-play, or refreshed by
// reanalyze); when state j+l's is older the first fresh one after it is used,
// and without one the target is the final result. horizon 0 always uses the
// final result, as MuZero does for board games.
struct ValueTargetConfig {
  int horizon = 0;
  int min_horizon = 1;
  int horizon_decay_steps = 500;
  int max_value_age = 1000;
};

// samples positions from a replay buffer and builds unrolled targets, each
// sample under a random symmetry of the encoder; step is the learner step the
// batch trains, which ages the value targets
template <typename Game>
struct BatchBuilder {
  BatchBuilder(const Encoder<Game>* encoder, int batch_size, int unroll_steps, bool augment,
               const ValueTargetConfig& value = ValueTargetConfig())
  : mEncoder(encoder), mBatchSize(batch_size), mUnroll(unroll_steps), mAugment(augment), mValue(value)
  {}

  void build(const ReplayBuffer& replay, s::mt19937_64& rng, TrainingBatch& batch, int64_t step = 0){
    int T = mEncoder->num_tokens();
    int F = mEncoder->num_features();
    int A = mEncoder->num_actions();
    int K = mUnroll;
    batch.resize(mBatchSize, K, T, F, A);
    replay.sample_positions(mBatchSize, rng, mRefs);
    s::uniform_int_distribution<int> sym_dist(0, mAugment ? mEncoder->num_symmetries() - 1 : 0);
    for (int b = 0; b < (int)mRefs.size(); ++b)
      fill(*mRefs[b].trajectory, mRefs[b].t, sym_dist(rng), b, batch, step);
  }

  // writes sample b of batch (already resized): position t of tr unrolled,
  // under symmetry sym
  void fill(const Trajectory& tr, int t, int sym, int b, TrainingBatch& batch, int64_t step = 0) const {
    int T = batch.T, F = batch.F, A = batch.A, K = batch.K;
    const s::vector<int>& tmap = mEncoder->token_map(sym);
    const s::vector<int>& amap = mEncoder->action_map(sym);

    auto copy_obs = [&](int state, uint8_t* dst){
      const uint8_t* src = tr.observation(state);
      for (int i = 0; i < T; ++i)
        s::copy(src + i * F, src + (i + 1) * F, dst + tmap[i] * F);
    };

    copy_obs(t, batch.observations.data() + (size_t)b * T * F);
    s::lock_guard<s::mutex> lock(tr.policy_mutex);
    for (int k = 0; k <= K; ++k){
      int j = t + k;
      size_t bk = (size_t)b * (K + 1) + k;
      if (j <= tr.length){
        batch.state_mask[bk] = 1.f;
        const uint8_t* tok = tr.token_classes(j);
        for (int i = 0; i < T; ++i)
          batch.tokens[bk * T + tmap[i]] = tok[i];
        if (k > 0) copy_obs(j, batch.target_observations.data() + ((size_t)b * K + k - 1) * T * F);
      }
      if (j < tr.length){
        batch.value_mask[bk] = 1.f;
        float v = value_target(tr, j, step);
        float* vt = batch.values.data() + bk * 3;
        vt[0] = s::max(-v, 0.f);
        vt[2] = s::max(v, 0.f);
        vt[1] = 1.f - vt[0] - vt[2];
        bool flip = tr.to_play[j] != 0;
        for (int i = 0; i < T; ++i){
          int c = tr.final_tokens[i];
          batch.ownership[bk * T + tmap[i]] = flip ? mEncoder->flip_token_class(c) : c;
        }
        if (tr.policy_valid[j]){
          batch.policy_mask[bk] = 1.f;
          const float* p = tr.policies.data() + (size_t)j * A;
          for (int a = 0; a < A; ++a)
            batch.policies[bk * A + amap[a]] = p[a];
        }
      }
      if (k < K){
        size_t bk2 = (size_t)b * K + k;
        if (j < tr.length){
          batch.actions[bk2] = amap[tr.actions[j]];
          batch.rewards[bk2] = tr.rewards[j] + 1;
          batch.reward_mask[bk2] = 1.f;
        } else {
          batch.actions[bk2] = 0;
        }
      }
    }
  }

  // value target of non-terminal state j for its side to move; the caller
  // holds tr.policy_mutex
  float value_target(const Trajectory& tr, int j, int64_t step) const {
    if (mValue.horizon <= 0) return (float)tr.value_target(j);
    int64_t age = s::max<int64_t>(0, step - (int64_t)tr.model_version);
    int64_t shrink = age / s::max(1, mValue.horizon_decay_steps);
    int l = (int)s::max<int64_t>(s::max(1, mValue.min_horizon), mValue.horizon - shrink);
    // rewards are for the player who moved
    float rewards = 0.f;
    for (int i = j; i < tr.length; ++i){
      if (i >= j + l){
        uint32_t version = tr.value_versions[i];
        if (version != Trajectory::NO_VALUE and step - (int64_t)version <= mValue.max_value_age){
          float boot = tr.to_play[i] == tr.to_play[j] ? tr.root_values[i] : -tr.root_values[i];
          return s::clamp(rewards + boot, -1.f, 1.f);
        }
      }
      rewards += tr.to_play[i] == tr.to_play[j] ? tr.rewards[i] : -tr.rewards[i];
    }
    return (float)tr.value_target(j);
  }

protected:
  const Encoder<Game>*  mEncoder;
  int                   mBatchSize;
  int                   mUnroll;
  bool                  mAugment;
  ValueTargetConfig     mValue;
  s::vector<PositionRef> mRefs;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_BATCH_H
