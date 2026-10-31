#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <evaluation/snapshot_evaluator.h>
#include <search/mcgs.h>

#include <common/othello_fixtures.h>
#include <common/temp_dir.h>

namespace mx = mlx::core;
namespace r = rlbg;
namespace rt = rlbg::test;
namespace sf = std::filesystem;

using Game = r::OthelloBitboard;

namespace {

int count_lines(const sf::path& p){
  std::ifstream in(p);
  int n = 0;
  for (std::string line; std::getline(in, line);) ++n;
  return n;
}

struct Fixture {
  r::OthelloEncoder enc;
  r::EvaluatorConfig cfg;
  rt::TempDir dir;

  Fixture(){
    cfg.games = 2;
    cfg.simulations = 2;
    cfg.baseline_games = 2;
    cfg.rollout_simulations = 10;
    cfg.gauntlet = {1};
  }

  std::unique_ptr<r::SnapshotEvaluator<Game>> evaluator(){
    int latent = rt::small_model(enc)->latent_size();
    auto make_search = [this, latent](uint64_t seed){
      return std::unique_ptr<r::Search<Game>>(std::make_unique<r::MCGS<Game>>(&enc, latent, r::MCGSConfig{}, seed));
    };
    auto make_model = [this]{ return std::shared_ptr<r::MuZeroModel>(rt::small_model(enc)); };
    return std::make_unique<r::SnapshotEvaluator<Game>>(
      cfg, (dir / "eval").string(), make_model, enc.observation_size(), mx::float32, make_search, 1);
  }

  std::string snapshot(const std::string& name){
    std::string file = (dir / (name + ".safetensors")).string();
    rt::small_model(enc)->save_weights(file);
    return file;
  }

  // evaluates one snapshot on the evaluator's thread
  void evaluate(r::SnapshotEvaluator<Game>& ev, const std::string& name){
    std::string before = ev.summary();
    std::atomic<bool> stop{false};
    std::thread t([&]{ ev.run(stop); });
    ev.submit(name, snapshot(name));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (ev.summary() == before and std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop = true;
    t.join();
  }
};

} // namespace

TEST(SnapshotEvaluator, RatesSnapshotsAgainstEachOtherAndBaselines){
  Fixture f;
  auto ev = f.evaluator();
  f.evaluate(*ev, "step_00000000");
  EXPECT_NE(ev->summary().find("step_00000000"), std::string::npos);
  // the first snapshot has no older snapshot: random and rollout baselines
  EXPECT_EQ(count_lines(f.dir / "eval/results.csv"), 3);
  f.evaluate(*ev, "step_00000010");
  EXPECT_NE(ev->summary().find("vs step_00000000"), std::string::npos) << ev->summary();
  EXPECT_EQ(count_lines(f.dir / "eval/results.csv"), 6);

  std::ifstream in(f.dir / "eval/ratings.csv");
  std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  for (const char* p : {"random,0.0", "step_00000000,", "step_00000010,", "mcts_rollout_10,"})
    EXPECT_NE(all.find(p), std::string::npos) << p;
}

TEST(SnapshotEvaluator, ReloadsResultsAndSnapshotsOnRestart){
  Fixture f;
  {
    auto ev = f.evaluator();
    f.evaluate(*ev, "step_00000000");
  }
  auto ev = f.evaluator();
  ev->add_existing("step_00000000", (f.dir / "step_00000000.safetensors").string());
  f.evaluate(*ev, "step_00000010");
  EXPECT_NE(ev->summary().find("vs step_00000000"), std::string::npos) << ev->summary();
  EXPECT_EQ(count_lines(f.dir / "eval/results.csv"), 6);
  std::ifstream in(f.dir / "eval/ratings.csv");
  std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(all.find("step_00000000,"), std::string::npos) << "earlier results are part of the fit";
}
