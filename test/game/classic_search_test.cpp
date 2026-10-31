// the minimax and rollout graph search used by the interactive apps
#include <gtest/gtest.h>

#include <types.h>
#include <connectx.h>
#include <othello.h>
#include <minimax_search.h>
#include <mcgs_search.h>

namespace r = rlbg;

namespace {

using TicTac = r::ConnectXGameState<3, 3, 3>;

template <typename Game>
void drops(Game& g, std::initializer_list<int> cols){
  for (int c : cols) ASSERT_TRUE(g.apply(r::PlayerMove(g.next_player(), r::Move(r::M::Play, r::Pt(0, c)))));
}

} // namespace

TEST(Minimax, TakesAnImmediateWin){
  // black has (0,0) (1,0); dropping in column 0 completes the column
  TicTac g;
  drops(g, {0, 1, 0, 2});
  EXPECT_EQ(r::minimax_search(g).mpt.c, 0);
  EXPECT_EQ(r::minimax_search_ab(g).mpt.c, 0);
}

TEST(Minimax, BlocksAnImmediateLoss){
  // white must stop black's column 2
  TicTac g;
  drops(g, {2, 0, 2});
  EXPECT_EQ(r::minimax_search(g).mpt.c, 2);
  EXPECT_EQ(r::minimax_search_ab(g).mpt.c, 2);
}

TEST(Minimax, SearchLeavesStateUnchanged){
  TicTac g;
  drops(g, {1});
  size_t h = g.hash();
  r::minimax_search_ab(g);
  EXPECT_EQ(g.hash(), h);
  EXPECT_EQ(g.history().size(), 1u);
}

TEST(RolloutMCGS, TakesAnImmediateWin){
  using Connect4 = r::ConnectXGameState<6, 7, 4>;
  using Board = r::ConnectXBoard<6, 7>;
  Connect4 g;
  drops(g, {0, 1, 0, 1, 0, 1});
  r::MCPlayout<Board, Connect4> playout(8);
  r::UCTSelector<Board, Connect4> selector(1.0);
  r::NullCollector<Board, Connect4> collector;
  r::Move m = r::mcgs_search(g, selector, playout, collector, 2000);
  EXPECT_EQ(m.mty, r::M::Play);
  EXPECT_EQ(m.mpt.c, 0);
}

TEST(RolloutMCGS, ReturnsALegalOthelloMove){
  using Game = r::OthelloGameState<8>;
  using Board = r::OthelloBoard<8>;
  Game g;
  r::MCPlayout<Board, Game> playout(4);
  r::UCTSelector<Board, Game> selector(0.5);
  r::NullCollector<Board, Game> collector;
  for (int i = 0; i < 4; ++i){
    r::Move m = r::mcgs_search(g, selector, playout, collector, 200);
    ASSERT_TRUE(g.is_valid(r::PlayerMove(g.next_player(), m))) << m;
    ASSERT_TRUE(g.apply(r::PlayerMove(g.next_player(), m)));
  }
}
