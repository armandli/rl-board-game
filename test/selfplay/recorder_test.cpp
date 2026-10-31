#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <search/mcgs.h>
#include <selfplay/recorder.h>

#include <common/othello_fixtures.h>
#include <common/random_games.h>
#include <common/true_model.h>

namespace r = rlbg;
namespace rt = rlbg::test;

TEST(TrajectoryRecorder, RecordsAConsistentGame){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  r::MCGSConfig cfg;
  auto games = rt::play_selfplay_games<r::OthelloBitboard>(4, model, enc, cfg, 8);
  for (const auto& tp : games){
    const r::Trajectory& t = *tp;
    int n = t.length;
    ASSERT_GT(n, 8);
    EXPECT_EQ(t.features.size(), (size_t)(n + 1) * enc.observation_size());
    EXPECT_EQ(t.tokens.size(), (size_t)(n + 1) * 64);
    EXPECT_EQ(t.to_play.size(), (size_t)n + 1);
    EXPECT_EQ(t.actions.size(), (size_t)n);
    EXPECT_EQ(t.rewards.size(), (size_t)n);
    EXPECT_EQ(t.root_values.size(), (size_t)n);
    EXPECT_EQ(t.policies.size(), (size_t)n * 65);
    EXPECT_EQ(t.policy_valid.size(), (size_t)n);
    EXPECT_EQ(t.final_tokens.size(), 64u);

    // replaying the actions reproduces every recorded state
    r::OthelloBitboard g;
    std::vector<uint8_t> obs(enc.observation_size());
    for (int s = 0; s <= n; ++s){
      enc.encode(g, obs.data());
      ASSERT_TRUE(std::equal(obs.begin(), obs.end(), t.observation(s))) << "state " << s;
      EXPECT_EQ(t.to_play[s], g.to_play());
      if (s == n) break;
      ASSERT_TRUE(g.is_legal(t.actions[s]));
      int mover = g.to_play();
      g.apply(t.actions[s]);
      EXPECT_EQ(t.rewards[s], g.is_terminal() ? g.outcome(mover) : 0);
    }
    EXPECT_TRUE(g.is_terminal());
    EXPECT_EQ(t.outcome, g.outcome(0));
    std::vector<uint8_t> fin(64);
    enc.encode_tokens_absolute(g, fin.data());
    EXPECT_EQ(t.final_tokens, fin);
  }
}
