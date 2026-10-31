#include <algorithm>
#include <bit>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>

#include <common/tictactoe.h>

namespace r = rlbg;
namespace rt = rlbg::test;

using Game = r::OthelloBitboard;

namespace {

// random positions from random games, terminal ones included
std::vector<Game> random_positions(int n, uint64_t seed){
  std::mt19937_64 rng(seed);
  std::vector<Game> out;
  while ((int)out.size() < n){
    Game g;
    r::ActionList<Game::NUM_ACTIONS> legal;
    while (not g.is_terminal() and (int)out.size() < n){
      out.push_back(g);
      g.legal_actions(legal);
      g.apply(legal[std::uniform_int_distribution<int>(0, legal.size - 1)(rng)]);
    }
    if ((int)out.size() < n) out.push_back(g);
  }
  return out;
}

uint64_t transform(uint64_t bits, const std::vector<int>& map){
  uint64_t out = 0;
  for (int i = 0; i < 64; ++i)
    if ((bits >> i) & 1) out |= 1ULL << map[i];
  return out;
}

} // namespace

TEST(OthelloEncoder, Dimensions){
  r::OthelloEncoder enc;
  EXPECT_EQ(enc.num_tokens(), 64);
  EXPECT_EQ(enc.num_features(), 4);
  EXPECT_EQ(enc.observation_size(), 256);
  EXPECT_EQ(enc.num_actions(), 65);
  EXPECT_EQ(enc.num_token_classes(), 3);
  EXPECT_EQ(enc.num_symmetries(), 8);
}

TEST(OthelloEncoder, FeaturesDescribeTheBoardFromTheMover){
  r::OthelloEncoder enc;
  std::vector<uint8_t> f(enc.observation_size());
  for (const Game& g : random_positions(500, 1)){
    enc.encode(g, f.data());
    for (int i = 0; i < 64; ++i){
      const uint8_t* t = f.data() + i * 4;
      ASSERT_EQ(t[0] + t[1] + t[2], 1) << "exactly one of empty, own, opponent";
      EXPECT_EQ(t[1], (g.own() >> i) & 1);
      EXPECT_EQ(t[2], (g.opp() >> i) & 1);
      uint64_t legal = g.is_terminal() ? 0 : g.legal();
      EXPECT_EQ(t[3], (legal >> i) & 1);
    }
  }
}

TEST(OthelloEncoder, TokenClasses){
  r::OthelloEncoder enc;
  std::vector<uint8_t> rel(64), abs(64);
  for (const Game& g : random_positions(300, 2)){
    enc.encode_tokens(g, rel.data());
    enc.encode_tokens_absolute(g, abs.data());
    for (int i = 0; i < 64; ++i){
      int expected = g.to_play() == 0 ? abs[i] : enc.flip_token_class(abs[i]);
      ASSERT_EQ(rel[i], expected);
      EXPECT_EQ(abs[i], ((g.black() >> i) & 1) ? 1 : ((g.white() >> i) & 1) ? 2 : 0);
    }
  }
  for (int c = 0; c < 3; ++c) EXPECT_EQ(enc.flip_token_class(enc.flip_token_class(c)), c);
  EXPECT_EQ(enc.flip_token_class(0), 0);
}

TEST(OthelloEncoder, SymmetriesFormTheDihedralGroup){
  r::OthelloEncoder enc;
  std::set<std::vector<int>> maps;
  for (int s = 0; s < 8; ++s){
    const std::vector<int>& m = enc.token_map(s);
    std::vector<int> sorted = m;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 64; ++i) ASSERT_EQ(sorted[i], i) << "symmetry " << s << " is a permutation";
    EXPECT_EQ(enc.action_map(s).size(), 65u);
    EXPECT_EQ(enc.action_map(s)[Game::PASS], Game::PASS);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(enc.action_map(s)[i], m[i]);
    maps.insert(m);
  }
  EXPECT_EQ(maps.size(), 8u);
  for (int i = 0; i < 64; ++i) EXPECT_EQ(enc.token_map(0)[i], i) << "symmetry 0 is the identity";
  // closed under composition
  for (int a = 0; a < 8; ++a)
    for (int b = 0; b < 8; ++b){
      std::vector<int> c(64);
      for (int i = 0; i < 64; ++i) c[i] = enc.token_map(a)[enc.token_map(b)[i]];
      EXPECT_TRUE(maps.contains(c)) << a << " after " << b;
    }
}

// the rules are invariant: legal moves of a transformed board are the
// transformed legal moves
TEST(OthelloEncoder, SymmetriesPreserveTheRules){
  r::OthelloEncoder enc;
  for (const Game& g : random_positions(300, 3))
    for (int s = 0; s < 8; ++s){
      const std::vector<int>& m = enc.token_map(s);
      uint64_t own = transform(g.own(), m);
      uint64_t opp = transform(g.opp(), m);
      EXPECT_EQ(Game::legal_moves(own, opp), transform(Game::legal_moves(g.own(), g.opp()), m));
      if (g.legal() == 0) continue;
      int a = std::countr_zero(g.legal());
      EXPECT_EQ(transform(Game::flips(g.own(), g.opp(), a), m), Game::flips(own, opp, m[a]));
    }
}

TEST(TicTacToeEncoder, SymmetriesPreserveWinningLines){
  rt::TicTacToeEncoder enc;
  for (int s = 0; s < 8; ++s)
    for (uint16_t line : rt::TicTacToe::LINES){
      uint16_t t = 0;
      for (int i = 0; i < 9; ++i)
        if ((line >> i) & 1) t |= 1 << enc.token_map(s)[i];
      EXPECT_TRUE(rt::TicTacToe::wins(t)) << "symmetry " << s;
    }
}
