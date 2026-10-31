#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <training/training_loop.h>

#include <common/othello_fixtures.h>
#include <common/temp_dir.h>

namespace r = rlbg;
namespace rt = rlbg::test;
namespace sf = std::filesystem;

using Game = r::OthelloBitboard;

namespace {

r::TrainingLoopConfig tiny(const std::string& dir, int max_steps){
  r::TrainingLoopConfig c;
  c.data_dir = dir;
  c.max_steps = max_steps;
  c.seed = 1;
  c.batch_size = 16;
  c.min_replay_positions = 200;
  c.selfplay_lead = 200;
  c.sample_reuse = 4.;
  c.publish_interval = 2;
  c.snapshot_interval = 6;
  c.checkpoint_interval = 4;
  c.num_actors = 3;
  c.reanalyze_actors = 1;
  c.log_interval_s = 1;
  c.selfplay.games_per_actor = 8;
  c.selfplay.num_simulations = 4;
  c.selfplay.fast_simulations = 2;
  c.evaluator.games = 2;
  c.evaluator.simulations = 2;
  c.evaluator.baseline_games = 2;
  c.evaluator.rollout_simulations = 10;
  c.learner.warmup_steps = 2;
  c.learner.total_steps = 100;
  return c;
}

int64_t checkpoint_step(const sf::path& data){
  int64_t step = -1;
  std::ifstream(data / "checkpoint/learner.txt") >> step;
  return step;
}

} // namespace

// the whole system on a tiny model: self-play, reanalyze, inference server,
// learner and evaluator, then a resumed run
TEST(TrainingLoop, TrainsSavesAndResumes){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  auto make_model = [&]{ return std::shared_ptr<r::MuZeroModel>(rt::small_model(enc)); };
  std::atomic<bool> interrupt{false};
  {
    r::TrainingLoop<Game> loop(tiny(dir.str(), 12), &enc, make_model);
    loop.run(interrupt);
  }
  EXPECT_EQ(checkpoint_step(dir.path), 12);
  for (const char* f : {"checkpoint/model.safetensors", "checkpoint/optimizer.safetensors", "checkpoint/loop.txt",
                        "snapshots/step_00000000.safetensors", "snapshots/step_00000006.safetensors",
                        "snapshots/step_00000012.safetensors", "experience/games_000000.bin", "train_log.csv"})
    EXPECT_TRUE(sf::exists(dir / f)) << f;
  uint64_t positions = 0;
  std::ifstream(dir / "checkpoint/loop.txt") >> positions;
  EXPECT_GE(positions, 200u);

  {
    r::TrainingLoop<Game> loop(tiny(dir.str(), 18), &enc, make_model);
    loop.run(interrupt);
  }
  EXPECT_EQ(checkpoint_step(dir.path), 18);
  EXPECT_TRUE(sf::exists(dir / "snapshots/step_00000018.safetensors"));
}

TEST(TrainingLoop, StopsOnInterrupt){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  auto make_model = [&]{ return std::shared_ptr<r::MuZeroModel>(rt::small_model(enc)); };
  r::TrainingLoopConfig c = tiny(dir.str(), 1000000);
  c.evaluate = false;
  std::atomic<bool> interrupt{true};
  r::TrainingLoop<Game> loop(c, &enc, make_model);
  loop.run(interrupt);
  // stopped right away, still with a checkpoint to resume from
  EXPECT_GE(checkpoint_step(dir.path), 0);
  EXPECT_LT(checkpoint_step(dir.path), 1000000);
}
