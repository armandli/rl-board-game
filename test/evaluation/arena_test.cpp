#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <evaluation/agents.h>
#include <evaluation/arena.h>
#include <search/mcgs.h>

#include <common/true_model.h>

namespace r = rlbg;
namespace rt = rlbg::test;

using Game = r::OthelloBitboard;

namespace {

// a random player that records which colors it was asked to move
struct RecordingAgent : r::RandomAgent<Game> {
  using r::RandomAgent<Game>::RandomAgent;
  std::string name() const override { return "recording"; }
  void select_actions(const std::vector<const Game*>& games, std::vector<int>& actions) override {
    for (const Game* g : games) colors.insert(g->to_play());
    calls += 1;
    r::RandomAgent<Game>::select_actions(games, actions);
    for (size_t i = 0; i < games.size(); ++i) EXPECT_TRUE(games[i]->is_legal(actions[i]));
  }
  std::set<int> colors;
  int calls = 0;
};

} // namespace

TEST(Arena, PlaysEveryGameWithBothColors){
  RecordingAgent a(1), b(2);
  std::mt19937_64 rng(3);
  r::MatchResult m = r::play_match<Game>(a, b, 10, 2, rng);
  EXPECT_EQ(m.games(), 10);
  EXPECT_EQ(a.colors, (std::set<int>{0, 1}));
  EXPECT_EQ(b.colors, (std::set<int>{0, 1}));
  // all games run at once: one call per agent turn, not per game
  EXPECT_LT(a.calls, 10 * 40);
}

TEST(Arena, OddGameCounts){
  r::RandomAgent<Game> a(1), b(2);
  std::mt19937_64 rng(4);
  EXPECT_EQ(r::play_match<Game>(a, b, 7, 4, rng).games(), 7);
}

TEST(Arena, ColorSwappedPairsCancelOutSymmetricPlay){
  // two copies of a deterministic agent: every opening is won once by each side
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  auto make = [&](uint64_t seed){
    return std::unique_ptr<r::Search<Game>>(std::make_unique<r::MCGS<Game>>(&enc, model.latent_size(), r::MCGSConfig{}, seed));
  };
  r::NetworkAgent<Game> a("a", &model, make, 8), b("b", &model, make, 8);
  std::mt19937_64 rng(5);
  r::MatchResult m = r::play_match<Game>(a, b, 12, 4, rng);
  EXPECT_EQ(m.wins, m.losses);
}

TEST(Agents, RolloutMCTSBeatsRandom){
  r::RolloutMCTSAgent<Game> mcts(200, 1);
  r::RandomAgent<Game> random(2);
  EXPECT_EQ(mcts.name(), "mcts_rollout_200");
  std::mt19937_64 rng(6);
  EXPECT_GT(r::play_match<Game>(mcts, random, 10, 2, rng).score(), 0.8);
}

TEST(Agents, NetworkAgentPlaysLegalMovesAndBeatsRandom){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  auto make = [&](uint64_t seed){
    return std::unique_ptr<r::Search<Game>>(std::make_unique<r::MCGS<Game>>(&enc, model.latent_size(), r::MCGSConfig{}, seed));
  };
  r::NetworkAgent<Game> agent("net", &model, make, 32);
  agent.set_name("renamed");
  EXPECT_EQ(agent.name(), "renamed");
  r::RandomAgent<Game> random(2);
  std::mt19937_64 rng(7);
  EXPECT_GT(r::play_match<Game>(agent, random, 10, 2, rng).score(), 0.8);
}
