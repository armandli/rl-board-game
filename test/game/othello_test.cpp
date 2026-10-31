#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <types.h>
#include <othello.h>

namespace r = rlbg;

using Game = r::OthelloGameState<8>;

namespace {

r::PlayerMove play(const Game& g, int r_, int c){
  return r::PlayerMove(g.next_player(), r::Move(r::M::Play, r::Pt(r_, c)));
}

std::vector<int> play_squares(const Game& g){
  std::vector<int> out;
  for (const r::Move& m : g.legal_moves())
    if (m.mty == r::M::Play) out.push_back(r::index<8>(m.mpt));
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace

TEST(Othello, InitialPosition){
  Game g;
  EXPECT_EQ(g.next_player(), r::Player::Black);
  EXPECT_FALSE(g.is_terminal());
  EXPECT_EQ(g.board().count_player_piece(r::Player::Black), 2u);
  EXPECT_EQ(g.board().count_player_piece(r::Player::White), 2u);
  EXPECT_EQ(play_squares(g), (std::vector<int>{19, 26, 37, 44}));
  // passing is always offered, as this engine's rules allow
  EXPECT_EQ(g.legal_moves().front(), r::Move(r::M::Pass));
}

TEST(Othello, RejectsInvalidMoves){
  Game g;
  EXPECT_FALSE(g.apply(play(g, 0, 0)));
  EXPECT_FALSE(g.apply(r::PlayerMove(r::Player::White, r::Move(r::M::Play, r::Pt(2, 3)))));
  EXPECT_TRUE(g.apply(play(g, 2, 3)));
  EXPECT_EQ(g.next_player(), r::Player::White);
  EXPECT_EQ(g.board()[r::Pt(3, 3)], r::Player::Black);
}

TEST(Othello, RollbackRestoresState){
  Game g;
  size_t h0 = g.hash();
  std::vector<r::PlayerMove> played;
  for (int i = 0; i < 6; ++i){
    // the last legal placement each turn
    r::PlayerMove pm(g.next_player(), g.legal_moves().back());
    ASSERT_EQ(pm.move.mty, r::M::Play);
    ASSERT_TRUE(g.apply(pm));
    played.push_back(pm);
  }
  size_t h6 = g.hash();
  Game copy = g;
  EXPECT_EQ(copy.hash(), h6);
  g.rollback(2);
  EXPECT_NE(g.hash(), h6);
  ASSERT_TRUE(g.apply(played[4]));
  ASSERT_TRUE(g.apply(played[5]));
  EXPECT_EQ(g.hash(), h6);
  g.rollback(6);
  EXPECT_EQ(g.hash(), h0);
  EXPECT_EQ(g.next_player(), r::Player::Black);
  EXPECT_EQ(play_squares(g), (std::vector<int>{19, 26, 37, 44}));
}

// regression: a run of opponent discs that reaches the edge without an
// anchoring disc must not flip
TEST(Othello, RunToEdgeDoesNotFlip){
  Game g;
  for (int a : {37, 29, 21, 30, 23})
    ASSERT_TRUE(g.apply(play(g, a / 8, a % 8)));
  ASSERT_EQ(g.next_player(), r::Player::White);
  ASSERT_TRUE(g.apply(play(g, 5, 4)));
  EXPECT_EQ(g.board()[r::Pt(4, 4)], r::Player::White);   // anchored by (3, 4)
  EXPECT_EQ(g.board()[r::Pt(4, 5)], r::Player::Black);   // run (4,5) (3,6) (2,7) leaves the board
  EXPECT_EQ(g.board()[r::Pt(3, 6)], r::Player::Black);
  EXPECT_EQ(g.board()[r::Pt(2, 7)], r::Player::Black);
}

TEST(Othello, PassThenResign){
  Game g;
  ASSERT_TRUE(g.apply(r::PlayerMove(r::Player::Black, r::Move(r::M::Pass))));
  // after a pass the opponent may resign instead of passing back
  EXPECT_EQ(g.legal_moves().front(), r::Move(r::M::Resign));
  ASSERT_TRUE(g.apply(r::PlayerMove(r::Player::White, r::Move(r::M::Resign))));
  EXPECT_TRUE(g.is_terminal());
  EXPECT_EQ(g.winner(), r::Player::Black);
  EXPECT_EQ(g.score(r::Player::Black), 1);
  EXPECT_EQ(g.score(r::Player::White), -1);
  EXPECT_TRUE(g.legal_moves().empty());
}
