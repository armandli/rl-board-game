#ifndef RLBG_TRAINING_BATCH_H
#define RLBG_TRAINING_BATCH_H

#include <cstdint>

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
struct TrainingBatch {
  int B = 0, K = 0, T = 0, F = 0, A = 0;

  s::vector<uint8_t> observations;        // B, T, F            state t
  s::vector<uint8_t> target_observations; // B, K, T, F         states t+1..t+K
  s::vector<int32_t> actions;             // B, K
  s::vector<float>   policies;            // B, K+1, A
  s::vector<float>   policy_mask;         // B, K+1
  s::vector<int32_t> values;              // B, K+1  outcome class
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
    values.assign((size_t)b * (k + 1), 1);
    value_mask.assign((size_t)b * (k + 1), 0.f);
    rewards.assign((size_t)b * k, 1);
    reward_mask.assign((size_t)b * k, 0.f);
    tokens.assign((size_t)b * (k + 1) * t, 0);
    ownership.assign((size_t)b * (k + 1) * t, 0);
    state_mask.assign((size_t)b * (k + 1), 0.f);
  }
};

// samples positions from a replay buffer and builds unrolled targets, each
// sample under a random symmetry of the encoder
template <typename Game>
struct BatchBuilder {
  BatchBuilder(const Encoder<Game>* encoder, int batch_size, int unroll_steps, bool augment)
  : mEncoder(encoder), mBatchSize(batch_size), mUnroll(unroll_steps), mAugment(augment)
  {}

  void build(const ReplayBuffer& replay, s::mt19937_64& rng, TrainingBatch& batch){
    int T = mEncoder->num_tokens();
    int F = mEncoder->num_features();
    int A = mEncoder->num_actions();
    int K = mUnroll;
    batch.resize(mBatchSize, K, T, F, A);
    replay.sample_positions(mBatchSize, rng, mRefs);
    s::uniform_int_distribution<int> sym_dist(0, mAugment ? mEncoder->num_symmetries() - 1 : 0);
    for (int b = 0; b < (int)mRefs.size(); ++b)
      fill(*mRefs[b].trajectory, mRefs[b].t, sym_dist(rng), b, batch);
  }

  // writes sample b of batch (already resized): position t of tr unrolled,
  // under symmetry sym
  void fill(const Trajectory& tr, int t, int sym, int b, TrainingBatch& batch) const {
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
        batch.values[bk] = tr.value_target(j) + 1;
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

protected:
  const Encoder<Game>*  mEncoder;
  int                   mBatchSize;
  int                   mUnroll;
  bool                  mAugment;
  s::vector<PositionRef> mRefs;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_BATCH_H
