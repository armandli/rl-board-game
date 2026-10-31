#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <encoder/othello_encoder.h>
#include <experience/binary_experience_store.h>

#include <common/random_games.h>
#include <common/temp_dir.h>

namespace r = rlbg;
namespace rt = rlbg::test;
namespace sf = std::filesystem;

namespace {

std::vector<std::shared_ptr<r::Trajectory>> games(int n, uint64_t seed){
  r::OthelloEncoder enc;
  std::mt19937_64 rng(seed);
  std::vector<std::shared_ptr<r::Trajectory>> out;
  for (int i = 0; i < n; ++i){
    out.push_back(rt::random_trajectory(enc, rng, 0.5));
    out.back()->model_version = 100 + i;
  }
  return out;
}

void expect_same(const r::Trajectory& a, const r::Trajectory& b){
  EXPECT_EQ(a.length, b.length);
  EXPECT_EQ(a.observation_size, b.observation_size);
  EXPECT_EQ(a.num_tokens, b.num_tokens);
  EXPECT_EQ(a.num_actions, b.num_actions);
  EXPECT_EQ(a.features, b.features);
  EXPECT_EQ(a.tokens, b.tokens);
  EXPECT_EQ(a.to_play, b.to_play);
  EXPECT_EQ(a.actions, b.actions);
  EXPECT_EQ(a.rewards, b.rewards);
  EXPECT_EQ(a.root_values, b.root_values);
  EXPECT_EQ(a.policies, b.policies);
  EXPECT_EQ(a.policy_valid, b.policy_valid);
  EXPECT_EQ(a.final_tokens, b.final_tokens);
  EXPECT_EQ(a.outcome, b.outcome);
  EXPECT_EQ(a.model_version, b.model_version);
}

int count_files(const sf::path& dir){
  int n = 0;
  for (const auto& e : sf::directory_iterator(dir)) n += e.path().extension() == ".bin";
  return n;
}

} // namespace

TEST(BinaryExperienceStore, RoundTripsEveryField){
  rt::TempDir dir;
  auto gs = games(5, 1);
  {
    r::BinaryExperienceStore store(dir.str());
    for (auto& g : gs) store.write(*g);
  }
  r::BinaryExperienceStore store(dir.str());
  auto back = store.load_recent(1u << 30);
  ASSERT_EQ(back.size(), gs.size());
  for (size_t i = 0; i < gs.size(); ++i) expect_same(*gs[i], *back[i]);
}

TEST(BinaryExperienceStore, RotatesFilesAndContinuesNumbering){
  rt::TempDir dir;
  auto gs = games(7, 2);
  {
    r::BinaryExperienceStore store(dir.str(), 3);
    for (auto& g : gs) store.write(*g);
  }
  EXPECT_EQ(count_files(dir.path), 3);
  {
    // a restarted store appends to new files instead of overwriting
    r::BinaryExperienceStore store(dir.str(), 3);
    store.write(*gs[0]);
  }
  EXPECT_EQ(count_files(dir.path), 4);
  r::BinaryExperienceStore store(dir.str(), 3);
  EXPECT_EQ(store.load_recent(1u << 30).size(), 8u);
}

TEST(BinaryExperienceStore, LoadRecentKeepsTheNewestPositions){
  rt::TempDir dir;
  auto gs = games(10, 3);
  r::BinaryExperienceStore store(dir.str(), 4);
  for (auto& g : gs) store.write(*g);
  size_t last3 = gs[7]->length + gs[8]->length + gs[9]->length;
  auto back = store.load_recent(last3);
  ASSERT_EQ(back.size(), 3u);
  EXPECT_EQ(back[0]->model_version, 107u);
  EXPECT_EQ(back[2]->model_version, 109u);
  // one position more needs one game more
  EXPECT_EQ(store.load_recent(last3 + 1).size(), 4u);
}

TEST(BinaryExperienceStore, IgnoresATruncatedLastRecord){
  rt::TempDir dir;
  auto gs = games(3, 4);
  {
    r::BinaryExperienceStore store(dir.str());
    for (auto& g : gs) store.write(*g);
  }
  sf::path file = dir / "games_000000.bin";
  sf::resize_file(file, sf::file_size(file) - 100);
  r::BinaryExperienceStore store(dir.str());
  auto back = store.load_recent(1u << 30);
  ASSERT_EQ(back.size(), 2u);
  expect_same(*gs[1], *back[1]);
}

TEST(BinaryExperienceStore, RejectsCorruptFiles){
  rt::TempDir dir;
  std::ofstream(dir / "games_000000.bin") << "not an experience file";
  r::BinaryExperienceStore store(dir.str());
  EXPECT_THROW(store.load_recent(100), std::runtime_error);
}
