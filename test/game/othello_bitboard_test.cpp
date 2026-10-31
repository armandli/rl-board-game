#include <algorithm>
#include <bit>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <types.h>
#include <othello.h>
#include <othello_bitboard.h>

namespace r = rlbg;

using BB = r::OthelloBitboard;

namespace {

std::vector<int> actions(const BB& g){
  r::ActionList<BB::NUM_ACTIONS> list;
  g.legal_actions(list);
  return std::vector<int>(list.begin(), list.end());
}

int random_action(const BB& g, std::mt19937_64& rng){
  std::vector<int> a = actions(g);
  return a[std::uniform_int_distribution<int>(0, (int)a.size() - 1)(rng)];
}

} // namespace

TEST(OthelloBitboard, InitialPosition){
  BB g;
  EXPECT_EQ(g.to_play(), 0);
  EXPECT_FALSE(g.is_terminal());
  EXPECT_EQ(std::popcount(g.black()), 2);
  EXPECT_EQ(std::popcount(g.white()), 2);
  EXPECT_EQ(actions(g), (std::vector<int>{19, 26, 37, 44}));
  EXPECT_FALSE(g.is_legal(BB::PASS));
  EXPECT_TRUE(g.is_legal(19));
  EXPECT_FALSE(g.is_legal(0));
}

TEST(OthelloBitboard, ApplyFlipsAndSwapsSides){
  BB g;
  g.apply(19);  // (2,3) flips (3,3)
  EXPECT_EQ(g.to_play(), 1);
  EXPECT_EQ(std::popcount(g.black()), 4);
  EXPECT_EQ(std::popcount(g.white()), 1);
  EXPECT_TRUE((g.black() >> 27) & 1);
  // own() is always the side to move
  EXPECT_EQ(g.own(), g.white());
  EXPECT_EQ(g.opp(), g.black());
}

TEST(OthelloBitboard, PassOnlyWithoutPlacements){
  std::mt19937_64 rng(3);
  int passes = 0;
  for (int game = 0; game < 300; ++game){
    BB g;
    while (not g.is_terminal()){
      std::vector<int> a = actions(g);
      bool has_pass = std::find(a.begin(), a.end(), BB::PASS) != a.end();
      EXPECT_EQ(has_pass, g.legal() == 0);
      if (has_pass){
        EXPECT_EQ(a.size(), 1u);
        ++passes;
      }
      for (int x : a) EXPECT_TRUE(g.is_legal(x));
      g.apply(random_action(g, rng));
    }
    EXPECT_TRUE(actions(g).empty());
    EXPECT_EQ(g.outcome(0), -g.outcome(1));
  }
  EXPECT_GT(passes, 0) << "random games should reach forced passes";
}

TEST(OthelloBitboard, HashSeparatesPositions){
  std::mt19937_64 rng(4);
  std::map<uint64_t, std::tuple<uint64_t, uint64_t, int>> seen;
  for (int game = 0; game < 200; ++game){
    BB g;
    while (not g.is_terminal()){
      auto key = std::make_tuple(g.black(), g.white(), g.to_play());
      auto [it, inserted] = seen.emplace(g.hash(), key);
      if (not inserted) EXPECT_EQ(it->second, key) << "hash collision";
      BB copy = g;
      EXPECT_EQ(copy.hash(), g.hash());
      g.apply(random_action(g, rng));
    }
  }
}

// plays random games on OthelloBitboard and the reference OthelloGameState
// side by side; they must agree on legal moves, boards and winners
TEST(OthelloBitboard, AgreesWithReferenceEngine){
  std::mt19937_64 rng(1234);
  for (int game = 0; game < 2000; ++game){
    r::OthelloGameState<8> ref;
    BB bb;
    while (not bb.is_terminal()){
      ASSERT_FALSE(ref.is_terminal()) << "game " << game;
      std::vector<int> ref_moves;
      for (const r::Move& m : ref.legal_moves())
        if (m.mty == r::M::Play) ref_moves.push_back(r::index<8>(m.mpt));
      std::sort(ref_moves.begin(), ref_moves.end());
      std::vector<int> bb_moves;
      for (int a : actions(bb))
        if (a != BB::PASS) bb_moves.push_back(a);
      ASSERT_EQ(ref_moves, bb_moves) << "game " << game;
      for (int i = 0; i < 64; ++i){
        r::Player p = ref.board()[r::point<8>(i)];
        r::Player q = (bb.black() >> i) & 1 ? r::Player::Black :
                      (bb.white() >> i) & 1 ? r::Player::White : r::Player::Unknown;
        ASSERT_EQ(p, q) << "game " << game << " square " << i;
      }
      int a = random_action(bb, rng);
      bb.apply(a);
      if (a == BB::PASS) ref.apply(r::PlayerMove(ref.next_player(), r::Move(r::M::Pass)));
      else               ref.apply(r::PlayerMove(ref.next_player(), r::Move(r::M::Play, r::point<8>(a))));
    }
    ASSERT_TRUE(ref.is_terminal());
    int o = bb.outcome(0);
    r::Player winner = o > 0 ? r::Player::Black : o < 0 ? r::Player::White : r::Player::Unknown;
    ASSERT_EQ(ref.winner(), winner) << "game " << game;
  }
}
