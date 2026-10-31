#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/fifo_replay_buffer.h>
#include <training/batch.h>
#include <training/learner.h>
#include <training/muzero_learner.h>

#include <common/othello_fixtures.h>
#include <common/random_games.h>
#include <common/temp_dir.h>

namespace mx = mlx::core;
namespace r = rlbg;
namespace rt = rlbg::test;

using Game = r::OthelloBitboard;

namespace {

r::TrainingBatch fixed_batch(const r::OthelloEncoder& enc, int size = 32){
  std::mt19937_64 rng(1);
  r::FifoReplayBuffer replay(100000);
  for (int i = 0; i < 8; ++i) replay.add(rt::random_trajectory(enc, rng, 0.8));
  r::BatchBuilder<Game> builder(&enc, size, 5, true);
  r::TrainingBatch batch;
  builder.build(replay, rng, batch);
  return batch;
}

r::MuZeroLearnerConfig config(const std::string& optimizer, const std::string& dtype){
  r::MuZeroLearnerConfig c;
  c.optimizer = optimizer;
  c.compute_dtype = dtype;
  c.warmup_steps = 5;
  c.total_steps = 1000;
  return c;
}

} // namespace

TEST(LearnerStatsAverage, AveragesAndResets){
  r::LearnerStatsAverage avg;
  r::LearnerStats a, b;
  a.loss = 2.; a.policy = 1.; a.learning_rate = 0.1; a.step = 1;
  b.loss = 4.; b.policy = 3.; b.learning_rate = 0.2; b.step = 2;
  avg.add(a);
  avg.add(b);
  EXPECT_EQ(avg.count(), 2);
  r::LearnerStats m = avg.take();
  EXPECT_DOUBLE_EQ(m.loss, 3.);
  EXPECT_DOUBLE_EQ(m.policy, 2.);
  EXPECT_DOUBLE_EQ(m.learning_rate, 0.2) << "the latest learning rate";
  EXPECT_EQ(m.step, 2);
  EXPECT_EQ(avg.count(), 0);
  EXPECT_DOUBLE_EQ(avg.take().loss, 0.);
}

class LearnerOverfits : public testing::TestWithParam<std::tuple<std::string, std::string>> {};

// every loss term falls on a fixed batch, whatever the optimizer and precision
TEST_P(LearnerOverfits, AFixedBatch){
  auto [optimizer, dtype] = GetParam();
  r::OthelloEncoder enc;
  r::TrainingBatch batch = fixed_batch(enc);
  r::MuZeroLearner learner(rt::small_model(enc), config(optimizer, dtype));
  r::LearnerStats first, last;
  for (int i = 0; i < 40; ++i){
    last = learner.train_step(batch);
    if (i == 0) first = last;
  }
  EXPECT_EQ(learner.step(), 40);
  EXPECT_TRUE(std::isfinite(last.loss));
  EXPECT_LT(last.loss, first.loss * 0.8);
  EXPECT_LT(last.policy, first.policy);
  EXPECT_LT(last.value, first.value);
  EXPECT_LT(last.reward, first.reward);
  EXPECT_LT(last.consistency, first.consistency);
  EXPECT_LT(last.reconstruction, first.reconstruction);
  EXPECT_LT(last.ownership, first.ownership);
  EXPECT_GT(last.grad_norm, 0.);
  EXPECT_GT(last.learning_rate, 0.);
}

INSTANTIATE_TEST_SUITE_P(OptimizersAndPrecisions, LearnerOverfits, testing::Combine(
  testing::Values("muon", "adamw"),
  testing::Values("float32", "bfloat16")
));

TEST(MuZeroLearner, WeightsAreTheModelsParametersInFloat32){
  r::OthelloEncoder enc;
  auto model = rt::small_model(enc);
  r::MuZeroLearner learner(model, config("muon", "bfloat16"));
  learner.train_step(fixed_batch(enc, 8));
  r::nn::Params w = learner.weights();
  EXPECT_EQ(w.size(), model->parameters().size());
  for (const auto& [k, v] : w) EXPECT_EQ(v.dtype(), mx::float32) << k;
  EXPECT_EQ(&learner.model(), model.get());
}

TEST(MuZeroLearner, CheckpointResumesExactly){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  r::TrainingBatch batch = fixed_batch(enc, 16);
  r::MuZeroLearner a(rt::small_model(enc), config("muon", "float32"));
  for (int i = 0; i < 3; ++i) a.train_step(batch);
  a.save_checkpoint(dir.str());

  r::MuZeroLearner b(rt::small_model(enc), config("muon", "float32"));
  ASSERT_TRUE(b.load_checkpoint(dir.str()));
  EXPECT_EQ(b.step(), 3);
  r::LearnerStats sa = a.train_step(batch);
  r::LearnerStats sb = b.train_step(batch);
  EXPECT_NEAR(sa.loss, sb.loss, 1e-4 * std::abs(sa.loss));
  EXPECT_NEAR(sa.learning_rate, sb.learning_rate, 1e-9);
}

TEST(MuZeroLearner, NoCheckpointInAnEmptyDirectory){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  r::MuZeroLearner learner(rt::small_model(enc), config("adamw", "float32"));
  EXPECT_FALSE(learner.load_checkpoint(dir.str()));
  EXPECT_EQ(learner.step(), 0);
}

TEST(MuZeroLearner, RejectsUnknownSettings){
  r::OthelloEncoder enc;
  EXPECT_THROW(r::MuZeroLearner(rt::small_model(enc), config("sgd", "float32")), std::invalid_argument);
  EXPECT_THROW(r::MuZeroLearner(rt::small_model(enc), config("adamw", "int8")), std::invalid_argument);
}
