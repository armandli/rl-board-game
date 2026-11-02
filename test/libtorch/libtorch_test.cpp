#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/fifo_replay_buffer.h>
#include <libtorch/learner.h>
#include <libtorch/model_inference.h>
#include <libtorch/training_backend.h>
#include <libtorch/transformer_muzero.h>
#include <training/basic_training_loop.h>
#include <training/batch.h>

#include <common/random_games.h>
#include <common/temp_dir.h>

namespace r = rlbg;
namespace lt = rlbg::libtorch;
namespace rt = rlbg::test;
namespace sf = std::filesystem;
namespace t = torch;

using Game = r::OthelloBitboard;

namespace {

// the tiny models gain nothing from many threads; keep the suite from
// taking the whole machine
const bool THREADS_SET = []{ t::set_num_threads(4); return true; }();

lt::TransformerMuZeroConfig tiny_config(const r::OthelloEncoder& enc){
  lt::TransformerMuZeroConfig c;
  c.num_tokens = enc.num_tokens();
  c.num_features = enc.num_features();
  c.num_actions = enc.num_actions();
  c.num_token_classes = enc.num_token_classes();
  c.dims = 32;
  c.num_heads = 4;
  c.repr_layers = 2;
  c.dyn_layers = 1;
  c.pred_layers = 1;
  c.proj_dims = 32;
  return c;
}

t::Tensor observations(const r::OthelloEncoder& enc, const std::vector<Game>& games){
  std::vector<uint8_t> f(games.size() * enc.observation_size());
  for (size_t i = 0; i < games.size(); ++i) enc.encode(games[i], f.data() + i * enc.observation_size());
  return t::from_blob(f.data(), {(int64_t)games.size(), enc.num_tokens(), enc.num_features()}, t::kUInt8).clone();
}

std::vector<Game> some_positions(int n, uint64_t seed){
  std::mt19937_64 rng(seed);
  std::vector<Game> out;
  r::ActionList<Game::NUM_ACTIONS> legal;
  while ((int)out.size() < n){
    Game g;
    int moves = std::uniform_int_distribution<int>(0, 30)(rng);
    for (int i = 0; i < moves and not g.is_terminal(); ++i){
      g.legal_actions(legal);
      g.apply(legal[std::uniform_int_distribution<int>(0, legal.size - 1)(rng)]);
    }
    if (not g.is_terminal()) out.push_back(g);
  }
  return out;
}

} // namespace

TEST(TransformerMuZero, OutputShapes){
  r::OthelloEncoder enc;
  lt::TransformerMuZero m(tiny_config(enc));
  t::Tensor obs = observations(enc, some_positions(3, 1));
  t::Tensor latent = m->represent(obs);
  EXPECT_EQ(latent.sizes(), (std::vector<int64_t>{3, 64, 32}));
  lt::DynamicsOutput d = m->dynamics(latent, t::tensor({0, 63, 64}, t::kLong));
  EXPECT_EQ(d.latent.sizes(), latent.sizes());
  EXPECT_EQ(d.reward_logits.sizes(), (std::vector<int64_t>{3, 3}));
  lt::Prediction p = m->predict(d.latent);
  EXPECT_EQ(p.policy_logits.sizes(), (std::vector<int64_t>{3, 65}));
  EXPECT_EQ(p.value_logits.sizes(), (std::vector<int64_t>{3, 3}));
  EXPECT_EQ(p.ownership_logits.sizes(), (std::vector<int64_t>{3, 64, 3}));
  EXPECT_EQ(m->reconstruct(latent).sizes(), (std::vector<int64_t>{3, 64, 3}));
  EXPECT_EQ(m->predict_projection(m->project(latent)).sizes(), (std::vector<int64_t>{3, 64, 32}));
  // the latent is normalized per token
  EXPECT_NEAR(latent.mean(-1).abs().max().item<float>(), 0.f, 1e-4);
}

TEST(TransformerMuZero, ActionChangesDynamics){
  r::OthelloEncoder enc;
  lt::TransformerMuZero m(tiny_config(enc));
  t::Tensor latent = m->represent(observations(enc, some_positions(1, 2))).repeat({2, 1, 1});
  lt::DynamicsOutput d = m->dynamics(latent, t::tensor({19, 26}, t::kLong));
  EXPECT_GT((d.latent[0] - d.latent[1]).abs().max().item<float>(), 1e-3);
}

TEST(TransformerMuZero, SaveLoadRoundTrip){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  lt::TransformerMuZeroConfig c = tiny_config(enc);
  lt::TransformerMuZero m(c);
  std::string file = (dir / "m.pt").string();
  lt::save_model(m, file);
  EXPECT_TRUE(lt::load_model_config(file) == c);
  lt::TransformerMuZero loaded = lt::load_model(file);
  t::Tensor obs = observations(enc, some_positions(4, 3));
  t::NoGradGuard ng;
  EXPECT_TRUE(t::allclose(m->predict(m->represent(obs)).policy_logits, loaded->predict(loaded->represent(obs)).policy_logits));
}

TEST(TransformerMuZero, CopyWeightsChecksConfiguration){
  r::OthelloEncoder enc;
  lt::TransformerMuZeroConfig c = tiny_config(enc);
  lt::TransformerMuZero a(c);
  c.repr_layers = 3;
  lt::TransformerMuZero b(c);
  EXPECT_THROW(lt::copy_weights(a, b), std::invalid_argument);
}

// a latent pool must give exactly the results of transferring latents
TEST(ModelInference, LatentPoolMatchesTransferredLatents){
  r::OthelloEncoder enc;
  lt::TransformerMuZeroConfig c = tiny_config(enc);
  lt::TransformerMuZero src(c);
  lt::ModelInference plain(c, t::kCPU, t::kFloat, 0);
  lt::ModelInference pooled(c, t::kCPU, t::kFloat, 100);
  plain.set_weights(src);
  pooled.set_weights(src);
  EXPECT_EQ(plain.latent_size(), 64 * 32);
  EXPECT_EQ(pooled.latent_size(), lt::ModelInference::LATENT_HANDLE_SIZE);

  std::vector<Game> games = some_positions(3, 4);
  auto initial = [&](r::Inference& inf, r::InferenceResults& res){
    r::InferenceRequests req;
    req.observation_size = inf.observation_size();
    req.latent_size = inf.latent_size();
    std::vector<uint8_t> f(enc.observation_size());
    for (const Game& g : games){
      enc.encode(g, f.data());
      req.add_initial(f.data());
    }
    inf.infer(req, res);
  };
  // two recurrent steps from each initial latent, mixed with a new initial request
  auto recurrent = [&](r::Inference& inf, const r::InferenceResults& prev, r::InferenceResults& res){
    r::InferenceRequests req;
    req.observation_size = inf.observation_size();
    req.latent_size = inf.latent_size();
    std::vector<uint8_t> f(enc.observation_size());
    enc.encode(games[0], f.data());
    req.add_initial(f.data());
    for (int i = 0; i < 3; ++i) req.add_recurrent(prev.latent(i), 10 + i);
    inf.infer(req, res);
  };
  r::InferenceResults a0, a1, b0, b1;
  initial(plain, a0);
  initial(pooled, b0);
  recurrent(plain, a0, a1);
  recurrent(pooled, b0, b1);
  for (const auto& [a, b] : {std::pair{&a0, &b0}, std::pair{&a1, &b1}}){
    ASSERT_EQ(a->values.size(), b->values.size());
    for (size_t i = 0; i < a->values.size(); ++i){
      EXPECT_NEAR(a->values[i], b->values[i], 2e-3);
      EXPECT_NEAR(a->rewards[i], b->rewards[i], 2e-3);
    }
    for (size_t i = 0; i < a->policy_logits.size(); ++i) EXPECT_NEAR(a->policy_logits[i], b->policy_logits[i], 2e-2);
  }
}

TEST(ModelInference, EvictedLatentThrows){
  r::OthelloEncoder enc;
  lt::TransformerMuZeroConfig c = tiny_config(enc);
  lt::ModelInference inf(c, t::kCPU, t::kFloat, 4);
  r::InferenceRequests req;
  req.observation_size = inf.observation_size();
  req.latent_size = inf.latent_size();
  std::vector<uint8_t> f(enc.observation_size());
  enc.encode(Game(), f.data());
  r::InferenceResults first, more;
  req.add_initial(f.data());
  inf.infer(req, first);
  std::vector<r::latent_t> handle(first.latents.begin(), first.latents.end());
  for (int i = 0; i < 3; ++i) inf.infer(req, more);
  // 4 latents made: the first is still in the pool of 4
  r::InferenceRequests rec;
  rec.observation_size = inf.observation_size();
  rec.latent_size = inf.latent_size();
  rec.add_recurrent(handle.data(), 19);
  EXPECT_NO_THROW(inf.infer(rec, more));
  // the recurrent call made a fifth, which replaced the first
  EXPECT_THROW(inf.infer(rec, more), std::runtime_error);
}

TEST(ModelInference, LoadsSnapshotFiles){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  lt::TransformerMuZeroConfig c = tiny_config(enc);
  lt::TransformerMuZero m(c);
  lt::save_model(m, (dir / "s.pt").string());
  lt::ModelInference a(c, t::kCPU, t::kFloat), b(c, t::kCPU, t::kFloat);
  a.set_weights(m);
  b.load_weights((dir / "s.pt").string());
  r::InferenceRequests req;
  req.observation_size = a.observation_size();
  req.latent_size = a.latent_size();
  std::vector<uint8_t> f(enc.observation_size());
  enc.encode(Game(), f.data());
  req.add_initial(f.data());
  r::InferenceResults ra, rb;
  a.infer(req, ra);
  b.infer(req, rb);
  EXPECT_FLOAT_EQ(ra.values[0], rb.values[0]);
}

class LearnerTest : public testing::TestWithParam<std::string> {};

// a fixed batch is fit: the loss falls, and a checkpoint restores the step,
// weights and optimizer so training continues identically
TEST_P(LearnerTest, FitsABatchAndRestoresCheckpoints){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  std::mt19937_64 rng(5);
  r::FifoReplayBuffer replay(100000);
  for (int i = 0; i < 20; ++i) replay.add(rt::random_trajectory(enc, rng));
  r::BatchBuilder<Game> builder(&enc, 16, 3, true);
  r::TrainingBatch batch;
  builder.build(replay, rng, batch);

  lt::LearnerConfig lc;
  lc.optimizer = GetParam();
  lc.warmup_steps = 1;
  lc.learning_rate = 3e-3;
  lc.muon_learning_rate = 0.02;
  t::manual_seed(1);
  lt::Learner learner(tiny_config(enc), lc, t::kCPU);
  r::LearnerStats first = learner.train_step(batch);
  for (int i = 0; i < 30; ++i) learner.train_step(batch);
  r::LearnerStats later = learner.train_step(batch);
  EXPECT_LT(later.loss, first.loss - 0.5) << first.loss << " -> " << later.loss;
  EXPECT_TRUE(std::isfinite(later.grad_norm));
  EXPECT_EQ(learner.step(), 32);

  learner.save_checkpoint((dir / "ckpt").string());
  r::LearnerStats next = learner.train_step(batch);
  lt::Learner restored(tiny_config(enc), lc, t::kCPU);
  ASSERT_TRUE(restored.load_checkpoint((dir / "ckpt").string()));
  EXPECT_EQ(restored.step(), 32);
  r::LearnerStats replayed = restored.train_step(batch);
  EXPECT_NEAR(replayed.loss, next.loss, 1e-3 * std::abs(next.loss));
  EXPECT_FALSE(restored.load_checkpoint((dir / "missing").string()));
}

INSTANTIATE_TEST_SUITE_P(Optimizers, LearnerTest, testing::Values("muon", "adamw"));

// K = 0 trains representation and prediction only
TEST(Learner, TrainsWithoutUnroll){
  r::OthelloEncoder enc;
  std::mt19937_64 rng(6);
  r::FifoReplayBuffer replay(100000);
  for (int i = 0; i < 5; ++i) replay.add(rt::random_trajectory(enc, rng));
  r::BatchBuilder<Game> builder(&enc, 8, 0, true);
  r::TrainingBatch batch;
  builder.build(replay, rng, batch);
  lt::Learner learner(tiny_config(enc), lt::LearnerConfig{}, t::kCPU);
  r::LearnerStats st = learner.train_step(batch);
  EXPECT_TRUE(std::isfinite(st.loss));
  EXPECT_EQ(st.consistency, 0.);
  EXPECT_EQ(st.reward, 0.);
}

TEST(Learner, ScheduleWarmsUpThenDecays){
  r::OthelloEncoder enc;
  lt::LearnerConfig lc;
  lc.warmup_steps = 10;
  lc.total_steps = 110;
  lc.final_lr_fraction = 0.1;
  lt::Learner learner(tiny_config(enc), lc, t::kCPU);
  EXPECT_NEAR(learner.schedule(0), 0.01, 1e-9);
  EXPECT_NEAR(learner.schedule(10), 1., 1e-9);
  EXPECT_NEAR(learner.schedule(60), 0.55, 1e-9);
  EXPECT_NEAR(learner.schedule(110), 0.1, 1e-9);
  EXPECT_NEAR(learner.schedule(1000), 0.1, 1e-9);
}

namespace {

lt::TrainingConfig tiny_run(const r::OthelloEncoder& enc, const std::string& dir, int max_steps){
  lt::TrainingConfig c;
  c.model = tiny_config(enc);
  c.device = "cpu";
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
  c.server.max_batch = 64;
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

// the whole system on a tiny model with the latent pool: self-play,
// reanalyze, inference server, learner and evaluator, then a resumed run
TEST(TrainingLoop, TrainsSavesAndResumes){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  std::atomic<bool> interrupt{false};
  {
    lt::TrainingConfig c = tiny_run(enc, dir.str(), 12);
    auto backend = std::make_shared<lt::TrainingBackend>(c);
    ASSERT_GT(backend->latent_pool(), 0);
    r::BasicTrainingLoop<Game> loop(c, &enc, backend);
    loop.run(interrupt);
  }
  EXPECT_EQ(checkpoint_step(dir.path), 12);
  for (const char* f : {"checkpoint/model.pt", "checkpoint/optimizer.pt", "checkpoint/loop.txt", "final_model.pt",
                        "snapshots/step_00000000.pt", "snapshots/step_00000006.pt", "snapshots/step_00000012.pt",
                        "experience/games_000000.bin", "train_log.csv", "eval/results.csv"})
    EXPECT_TRUE(sf::exists(dir / f)) << f;
  uint64_t positions = 0;
  std::ifstream(dir / "checkpoint/loop.txt") >> positions;
  EXPECT_GE(positions, 200u);

  {
    lt::TrainingConfig c = tiny_run(enc, dir.str(), 18);
    r::BasicTrainingLoop<Game> loop(c, &enc, std::make_shared<lt::TrainingBackend>(c));
    loop.run(interrupt);
  }
  EXPECT_EQ(checkpoint_step(dir.path), 18);
  EXPECT_TRUE(sf::exists(dir / "snapshots/step_00000018.pt"));
  EXPECT_TRUE(lt::load_model_config((dir / "final_model.pt").string()) == tiny_config(enc));
}

TEST(TrainingLoop, StopsAtTheTimeLimit){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  lt::TrainingConfig c = tiny_run(enc, dir.str(), 1000000);
  c.evaluate = false;
  c.max_minutes = 0.25;
  std::atomic<bool> interrupt{false};
  r::BasicTrainingLoop<Game> loop(c, &enc, std::make_shared<lt::TrainingBackend>(c));
  loop.run(interrupt);
  EXPECT_GE(checkpoint_step(dir.path), 0);
  EXPECT_LT(checkpoint_step(dir.path), 1000000);
  EXPECT_TRUE(sf::exists(dir / "final_model.pt"));
}
