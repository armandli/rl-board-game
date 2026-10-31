#include <cmath>

#include <gtest/gtest.h>

#include <evaluation/elo.h>

namespace r = rlbg;

TEST(EloTable, AnchorIsFixed){
  r::EloTable t;
  t.add("a", "b", r::MatchResult{3, 0, 1});
  auto e = t.ratings("b", 100.);
  EXPECT_DOUBLE_EQ(e["b"], 100.);
  EXPECT_GT(e["a"], 100.);
}

TEST(EloTable, EvenResultsGiveEqualRatings){
  r::EloTable t;
  t.add("a", "b", r::MatchResult{5, 2, 5});
  auto e = t.ratings("a");
  EXPECT_NEAR(e["b"], 0., 1e-6);
}

TEST(EloTable, ScoreMapsToTheEloDifference){
  r::EloTable t;
  // 3 wins in 4 plus one virtual draw: expected score 3.5 / 5 = 0.7
  t.add("a", "b", r::MatchResult{3, 0, 1});
  auto e = t.ratings("b");
  EXPECT_NEAR(e["a"], 400. * std::log10(0.7 / 0.3), 0.5);
}

TEST(EloTable, UnbeatenPlayersStayFinite){
  r::EloTable t;
  t.add("a", "b", r::MatchResult{50, 0, 0});
  auto e = t.ratings("b");
  EXPECT_TRUE(std::isfinite(e["a"]));
  EXPECT_GT(e["a"], 400.);
}

TEST(EloTable, OrdersPlayersByStrength){
  r::EloTable t;
  t.add("a", "b", r::MatchResult{8, 0, 2});
  t.add("b", "c", r::MatchResult{8, 0, 2});
  t.add("a", "c", r::MatchResult{9, 0, 1});
  auto e = t.ratings("c");
  EXPECT_GT(e["a"], e["b"]);
  EXPECT_GT(e["b"], e["c"]);
  // results add up over repeated matches
  r::EloTable u;
  u.add("a", "b", r::MatchResult{4, 0, 1});
  u.add("a", "b", r::MatchResult{4, 0, 1});
  r::EloTable v;
  v.add("a", "b", r::MatchResult{8, 0, 2});
  EXPECT_NEAR(u.ratings("b")["a"], v.ratings("b")["a"], 1e-6);
}

TEST(MatchResult, Score){
  EXPECT_EQ((r::MatchResult{1, 2, 1}).games(), 4);
  EXPECT_DOUBLE_EQ((r::MatchResult{1, 2, 1}).score(), 0.5);
  EXPECT_DOUBLE_EQ((r::MatchResult{3, 0, 1}).score(), 0.75);
  EXPECT_DOUBLE_EQ(r::MatchResult{}.score(), 0.5);
}
