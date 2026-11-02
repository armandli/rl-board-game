#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <thread>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/fifo_replay_buffer.h>
#include <search/mcgs.h>
#include <selfplay/actor.h>

#include <common/random_games.h>
#include <common/true_model.h>

namespace r = rlbg;
namespace rt = rlbg::test;

using Game = r::OthelloBitboard;

namespace {

r::SearchFactory<Game> factory(const r::OthelloEncoder& enc, int latent_size){
  return [&enc, latent_size](uint64_t seed){
    return std::unique_ptr<r::Search<Game>>(std::make_unique<r::MCGS<Game>>(&enc, latent_size, r::MCGSConfig{}, seed));
  };
}

r::SelfPlayConfig small_selfplay(){
  r::SelfPlayConfig c;
  c.games_per_actor = 4;
  c.num_simulations = 8;
  c.fast_simulations = 2;
  c.full_search_prob = 0.5f;
  c.reanalyze_positions = 8;
  return c;
}

// runs fn on a thread until done() or the timeout, then raises stop
template <typename F, typename D>
void run_until(F fn, D done, std::atomic<bool>& stop, int timeout_s = 60){
  std::thread t(fn);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
  while (not done() and std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  stop = true;
  t.join();
}

} // namespace

TEST(SelfPlayActor, FillsTheReplayBufferWithValidGames){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  r::FifoReplayBuffer replay(100000);
  r::SelfPlayStats stats;
  std::atomic<bool> stop{false};
  r::SelfPlayActor<Game> actor(small_selfplay(), &enc, &model, factory(enc, model.latent_size()),
                               &replay, nullptr, &stats, []{ return 7u; }, 1);
  run_until([&]{ actor.run(stop); }, [&]{ return replay.num_games() >= 6; }, stop);

  ASSERT_GE(replay.num_games(), 6u);
  EXPECT_EQ(stats.games.load(), replay.total_games());
  EXPECT_EQ(stats.positions.load(), replay.total_positions());
  EXPECT_GE(stats.moves.load(), stats.positions.load());
  EXPECT_EQ(stats.wins + stats.losses + stats.draws, stats.games.load());
  EXPECT_GT(stats.full_searches.load(), 0u);
  std::mt19937_64 rng(1);
  int valid = 0, fast = 0;
  for (int i = 0; i < 20; ++i){
    auto t = replay.sample_trajectory(rng);
    EXPECT_EQ(t->model_version, 7u);
    Game g = rt::replay_to(*t, 0);
    for (int s = 0; s < t->length; ++s){
      ASSERT_TRUE(g.is_legal(t->actions[s]));
      double sum = 0.;
      for (int a = 0; a < 65; ++a) sum += t->policies[s * 65 + a];
      EXPECT_NEAR(sum, 1., 1e-4);
      (t->policy_valid[s] ? valid : fast) += 1;
      g.apply(t->actions[s]);
    }
    EXPECT_TRUE(g.is_terminal());
  }
  // playout cap randomization records both kinds of moves
  EXPECT_GT(valid, 0);
  EXPECT_GT(fast, 0);
}

TEST(SelfPlayActor, PausesWhenAsked){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  r::FifoReplayBuffer replay(100000);
  r::SelfPlayStats stats;
  std::atomic<bool> stop{false};
  r::SelfPlayActor<Game> actor(small_selfplay(), &enc, &model, factory(enc, model.latent_size()),
                               &replay, nullptr, &stats, []{ return 0u; }, 2);
  std::atomic<bool> paused{true};
  actor.set_pause([&]{ return paused.load(); });
  std::thread t([&]{ actor.run(stop); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(stats.moves.load(), 0u);
  EXPECT_EQ(model.calls.load(), 0);
  paused = false;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (stats.moves.load() == 0 and std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  stop = true;
  t.join();
  EXPECT_GT(stats.moves.load(), 0u);
}

TEST(ReanalyzeActor, RefreshesPolicyTargetsAndValues){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  r::FifoReplayBuffer replay(100000);
  std::mt19937_64 rng(3);
  std::vector<std::shared_ptr<r::Trajectory>> games;
  for (int i = 0; i < 4; ++i){
    games.push_back(rt::random_trajectory(enc, rng, 0.0));
    replay.add(games.back());
  }
  r::SelfPlayStats stats;
  std::atomic<bool> stop{false};
  r::ReanalyzeActor<Game> actor(small_selfplay(), &model, factory(enc, model.latent_size()), &replay, &stats,
                                []{ return 9u; }, 4);
  run_until([&]{ actor.run(stop); }, [&]{ return stats.reanalyzed.load() >= 40; }, stop);

  ASSERT_GE(stats.reanalyzed.load(), 40u);
  int refreshed = 0;
  for (const auto& t : games){
    std::lock_guard<std::mutex> lock(t->policy_mutex);
    for (int s = 0; s < t->length; ++s){
      if (not t->policy_valid[s]) continue;
      ++refreshed;
      Game g = rt::replay_to(*t, s);
      // a searched position also gets a root value tagged with the
      // reanalyzing network's version; a forced move is not searched
      r::ActionList<Game::NUM_ACTIONS> legal;
      g.legal_actions(legal);
      if (legal.size > 1){
        EXPECT_EQ(t->value_versions[s], 9u);
        EXPECT_LE(std::abs(t->root_values[s]), 1.f);
      } else {
        EXPECT_EQ(t->value_versions[s], r::Trajectory::NO_VALUE);
      }
      double sum = 0.;
      for (int a = 0; a < 65; ++a){
        float p = t->policies[s * 65 + a];
        if (p > 0.f) EXPECT_TRUE(g.is_legal(a));
        sum += p;
      }
      EXPECT_NEAR(sum, 1., 1e-4);
    }
  }
  EXPECT_GE(refreshed, 30);
}

TEST(ReanalyzeActor, WaitsForAnEmptyBuffer){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  r::FifoReplayBuffer replay(1000);
  r::SelfPlayStats stats;
  std::atomic<bool> stop{false};
  r::ReanalyzeActor<Game> actor(small_selfplay(), &model, factory(enc, model.latent_size()), &replay, &stats,
                                []{ return 0u; }, 5);
  std::thread t([&]{ actor.run(stop); });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  stop = true;
  t.join();
  EXPECT_EQ(stats.reanalyzed.load(), 0u);
  EXPECT_EQ(model.calls.load(), 0);
}
