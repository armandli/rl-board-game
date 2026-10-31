#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/fifo_replay_buffer.h>
#include <training/batch.h>

#include <common/random_games.h>

namespace r = rlbg;
namespace rt = rlbg::test;

using Game = r::OthelloBitboard;

namespace {

constexpr int K = 5;

struct Fixture {
  r::OthelloEncoder enc;
  r::BatchBuilder<Game> builder{&enc, 1, K, true};
  r::TrainingBatch batch;
  std::mt19937_64 rng{1};

  Fixture(){ batch.resize(1, K, 64, 4, 65); }
};

} // namespace

TEST(BatchBuilder, IdentitySampleCopiesTheTrajectory){
  Fixture f;
  auto t = rt::random_trajectory(f.enc, f.rng);
  ASSERT_GT(t->length, 3 + K + 1);
  f.builder.fill(*t, 3, 0, 0, f.batch);
  const r::TrainingBatch& b = f.batch;
  EXPECT_TRUE(std::equal(b.observations.begin(), b.observations.end(), t->observation(3)));
  for (int k = 0; k <= K; ++k){
    int j = 3 + k;
    EXPECT_EQ(b.state_mask[k], 1.f);
    EXPECT_EQ(b.value_mask[k], 1.f);
    EXPECT_EQ(b.policy_mask[k], 1.f);
    EXPECT_EQ(b.values[k], t->value_target(j) + 1);
    for (int a = 0; a < 65; ++a) EXPECT_EQ(b.policies[k * 65 + a], t->policies[j * 65 + a]);
    for (int i = 0; i < 64; ++i){
      EXPECT_EQ(b.tokens[k * 64 + i], t->token_classes(j)[i]);
      int c = t->final_tokens[i];
      EXPECT_EQ(b.ownership[k * 64 + i], t->to_play[j] == 0 ? c : f.enc.flip_token_class(c));
    }
    if (k > 0)
      EXPECT_TRUE(std::equal(t->observation(j), t->observation(j) + 256, b.target_observations.begin() + (k - 1) * 256));
    if (k < K){
      EXPECT_EQ(b.actions[k], t->actions[j]);
      EXPECT_EQ(b.rewards[k], t->rewards[j] + 1);
      EXPECT_EQ(b.reward_mask[k], 1.f);
    }
  }
}

TEST(BatchBuilder, MasksStopAtTheEndOfTheGame){
  Fixture f;
  auto t = rt::random_trajectory(f.enc, f.rng);
  int L = t->length;
  f.builder.fill(*t, L - 2, 0, 0, f.batch);
  const r::TrainingBatch& b = f.batch;
  // states L-2, L-1 are playable, L is terminal, beyond does not exist
  for (int k = 0; k <= K; ++k){
    int j = L - 2 + k;
    EXPECT_EQ(b.value_mask[k], j < L ? 1.f : 0.f) << k;
    EXPECT_EQ(b.policy_mask[k], j < L ? 1.f : 0.f) << k;
    EXPECT_EQ(b.state_mask[k], j <= L ? 1.f : 0.f) << k;
    if (k < K){
      EXPECT_EQ(b.reward_mask[k], j < L ? 1.f : 0.f) << k;
      if (j >= L) EXPECT_EQ(b.actions[k], 0) << "padding action";
    }
  }
  // the move into the terminal state is rewarded with the result for its mover
  int mover = t->to_play[L - 1];
  int result = mover == 0 ? t->outcome : -t->outcome;
  EXPECT_EQ(b.rewards[1], result + 1);
  // the terminal observation is still a consistency target
  EXPECT_TRUE(std::equal(t->observation(L), t->observation(L) + 256, b.target_observations.begin() + 256));
}

TEST(BatchBuilder, FastSearchesAreNotPolicyTargets){
  Fixture f;
  auto t = rt::random_trajectory(f.enc, f.rng, 0.0);
  f.builder.fill(*t, 0, 0, 0, f.batch);
  for (int k = 0; k <= K; ++k){
    EXPECT_EQ(f.batch.policy_mask[k], 0.f);
    EXPECT_EQ(f.batch.value_mask[k], 1.f);
  }
}

// under every symmetry the sample stays a consistent othello sample: the
// policy covers legal squares of the transformed board, token classes agree
// with the features, and actions lead to the transformed next states
TEST(BatchBuilder, SymmetriesTransformEveryTargetAlike){
  Fixture f;
  auto t = rt::random_trajectory(f.enc, f.rng);
  for (int sym = 0; sym < 8; ++sym){
    f.builder.fill(*t, 5, sym, 0, f.batch);
    const r::TrainingBatch& b = f.batch;
    const std::vector<int>& m = f.enc.token_map(sym);
    for (int k = 0; k <= K; ++k){
      const uint8_t* obs = k == 0 ? b.observations.data() : b.target_observations.data() + (k - 1) * 256;
      for (int i = 0; i < 64; ++i){
        int c = b.tokens[k * 64 + i];
        ASSERT_EQ(obs[i * 4 + c], 1) << "token class matches features, sym " << sym;
        if (k < K and b.policies[k * 65 + i] > 0.f) ASSERT_EQ(obs[i * 4 + 3], 1) << "policy on a legal square";
      }
      for (int i = 0; i < 64; ++i)
        ASSERT_EQ(b.tokens[k * 64 + m[i]], t->token_classes(5 + k)[i]);
      if (k < K) ASSERT_EQ(b.actions[k], f.enc.action_map(sym)[t->actions[5 + k]]);
    }
  }
}

TEST(BatchBuilder, BuildsFromReplay){
  r::OthelloEncoder enc;
  std::mt19937_64 rng(2);
  r::FifoReplayBuffer replay(100000);
  for (int i = 0; i < 10; ++i) replay.add(rt::random_trajectory(enc, rng, 0.7));
  r::BatchBuilder<Game> builder(&enc, 32, K, true);
  r::TrainingBatch b;
  builder.build(replay, rng, b);
  EXPECT_EQ(b.B, 32);
  EXPECT_EQ(b.observations.size(), 32u * 64 * 4);
  for (int s = 0; s < 32; ++s){
    EXPECT_EQ(b.value_mask[s * (K + 1)], 1.f) << "the sampled position is never terminal";
    for (int k = 0; k <= K; ++k){
      size_t i = s * (K + 1) + k;
      if (b.value_mask[i] > 0) EXPECT_EQ(b.state_mask[i], 1.f);
      if (b.policy_mask[i] > 0) EXPECT_EQ(b.value_mask[i], 1.f);
      if (k < K) EXPECT_EQ(b.reward_mask[s * K + k], b.state_mask[i + 1]);
    }
  }
}
