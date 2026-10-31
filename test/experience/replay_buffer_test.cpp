#include <atomic>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <encoder/othello_encoder.h>
#include <experience/fifo_replay_buffer.h>
#include <experience/trajectory.h>

#include <common/random_games.h>

namespace r = rlbg;
namespace rt = rlbg::test;

namespace {

std::shared_ptr<r::Trajectory> of_length(int n){
  auto t = std::make_shared<r::Trajectory>();
  t->length = n;
  return t;
}

} // namespace

TEST(Trajectory, ValueTargetIsTheResultForTheMover){
  r::OthelloEncoder enc;
  std::mt19937_64 rng(1);
  for (int i = 0; i < 20; ++i){
    auto t = rt::random_trajectory(enc, rng);
    for (int s = 0; s < t->length; ++s){
      r::OthelloBitboard g = rt::replay_to(*t, s);
      EXPECT_EQ(t->to_play[s], g.to_play());
      r::OthelloBitboard end = rt::replay_to(*t, t->length);
      EXPECT_EQ(t->value_target(s), end.outcome(g.to_play()));
    }
  }
}

TEST(FifoReplayBuffer, EmptyBuffer){
  r::FifoReplayBuffer rb(100);
  std::mt19937_64 rng(1);
  std::vector<r::PositionRef> refs;
  rb.sample_positions(10, rng, refs);
  EXPECT_TRUE(refs.empty());
  EXPECT_EQ(rb.sample_trajectory(rng), nullptr);
  EXPECT_EQ(rb.num_positions(), 0u);
}

TEST(FifoReplayBuffer, CountsAndIds){
  r::FifoReplayBuffer rb(1000);
  for (int n : {10, 20, 30}) rb.add(of_length(n));
  EXPECT_EQ(rb.num_games(), 3u);
  EXPECT_EQ(rb.num_positions(), 60u);
  EXPECT_EQ(rb.total_positions(), 60u);
  EXPECT_EQ(rb.total_games(), 3u);
}

TEST(FifoReplayBuffer, EvictsOldestBeyondCapacity){
  r::FifoReplayBuffer rb(50);
  std::vector<std::shared_ptr<r::Trajectory>> ts;
  for (int i = 0; i < 10; ++i){
    ts.push_back(of_length(20));
    rb.add(ts.back());
  }
  // keeps the fewest newest games holding at least `capacity` positions
  EXPECT_EQ(rb.num_positions(), 60u);
  EXPECT_EQ(rb.num_games(), 3u);
  EXPECT_EQ(rb.total_positions(), 200u);
  EXPECT_EQ(rb.total_games(), 10u);
  std::mt19937_64 rng(2);
  for (int i = 0; i < 50; ++i) EXPECT_GE(rb.sample_trajectory(rng)->id, 7u);
  EXPECT_EQ(ts[9]->id, 9u);
}

TEST(FifoReplayBuffer, SamplesPositionsUniformly){
  r::FifoReplayBuffer rb(10000);
  rb.add(of_length(10));
  rb.add(of_length(30));
  rb.add(of_length(60));
  std::mt19937_64 rng(3);
  std::vector<r::PositionRef> refs;
  std::map<uint64_t, int> per_game;
  int n = 0;
  for (int k = 0; k < 200; ++k){
    rb.sample_positions(100, rng, refs);
    ASSERT_EQ(refs.size(), 100u);
    for (const auto& ref : refs){
      ASSERT_GE(ref.t, 0);
      ASSERT_LT(ref.t, ref.trajectory->length);
      ++per_game[ref.trajectory->id];
      ++n;
    }
  }
  // each game is drawn in proportion to its length
  EXPECT_NEAR(per_game[0] / (double)n, 0.1, 0.02);
  EXPECT_NEAR(per_game[1] / (double)n, 0.3, 0.02);
  EXPECT_NEAR(per_game[2] / (double)n, 0.6, 0.02);
}

TEST(FifoReplayBuffer, ConcurrentAddAndSample){
  r::FifoReplayBuffer rb(5000);
  std::atomic<bool> done{false};
  std::vector<std::thread> threads;
  for (int w = 0; w < 4; ++w)
    threads.emplace_back([&]{
      for (int i = 0; i < 500; ++i) rb.add(of_length(1 + i % 60));
    });
  std::atomic<int> bad{0};
  std::thread reader([&]{
    std::mt19937_64 rng(4);
    std::vector<r::PositionRef> refs;
    while (not done){
      rb.sample_positions(32, rng, refs);
      for (const auto& ref : refs)
        if (ref.t < 0 or ref.t >= ref.trajectory->length) ++bad;
    }
  });
  for (auto& t : threads) t.join();
  done = true;
  reader.join();
  EXPECT_EQ(bad.load(), 0);
  EXPECT_EQ(rb.total_games(), 2000u);
  EXPECT_GE(rb.num_positions(), 5000u);
}
