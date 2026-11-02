#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <types.h>
#include <connectx.h>

namespace r = rlbg;

namespace {

template <typename Game>
r::PlayerMove drop(const Game& g, int col){
  return r::PlayerMove(g.next_player(), r::Move(r::M::Play, r::Pt(0, col)));
}

template <typename Game>
void drops(Game& g, std::initializer_list<int> cols){
  for (int c : cols) ASSERT_TRUE(g.apply(drop(g, c))) << "column " << c;
}

// winner by brute force over every line of X; Unknown when there is none
template <ubyte R, ubyte C, ubyte X>
r::Player reference_winner(const r::ConnectXBoard<R, C>& b){
  const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  for (int row = 0; row < R; ++row)
    for (int col = 0; col < C; ++col){
      r::Player p = b[r::Pt(row, col)];
      if (p == r::Player::Unknown) continue;
      for (auto& d : dirs){
        int n = 0;
        for (int k = 0; k < X; ++k){
          int rr = row + d[0] * k, cc = col + d[1] * k;
          if (rr < 0 or rr >= R or cc < 0 or cc >= C or b[r::Pt(rr, cc)] != p) break;
          ++n;
        }
        if (n == X) return p;
      }
    }
  return r::Player::Unknown;
}

} // namespace

using Connect4 = r::ConnectXGameState<6, 7, 4>;

TEST(ConnectX, VerticalWin){
  Connect4 g;
  drops(g, {0, 1, 0, 1, 0, 1});
  EXPECT_FALSE(g.is_terminal());
  drops(g, {0});
  EXPECT_TRUE(g.is_terminal());
  EXPECT_EQ(g.winner(), r::Player::Black);
  EXPECT_EQ(g.score(r::Player::Black), 1);
  EXPECT_TRUE(g.legal_moves().empty());
}

TEST(ConnectX, HorizontalWin){
  Connect4 g;
  drops(g, {0, 0, 1, 1, 2, 2});
  drops(g, {3});
  EXPECT_TRUE(g.is_terminal());
  EXPECT_EQ(g.winner(), r::Player::Black);
}

TEST(ConnectX, DiagonalWins){
  // rising diagonal (0,0) (1,1) (2,2) (3,3) for black
  Connect4 up;
  drops(up, {0, 1, 1, 2, 2, 3, 2, 3, 3, 6});
  EXPECT_FALSE(up.is_terminal());
  drops(up, {3});
  EXPECT_TRUE(up.is_terminal());
  EXPECT_EQ(up.winner(), r::Player::Black);
  // falling diagonal (3,0) (2,1) (1,2) (0,3) for black
  Connect4 down;
  drops(down, {3, 2, 2, 1, 1, 0, 1, 0, 0, 6});
  EXPECT_FALSE(down.is_terminal());
  drops(down, {0});
  EXPECT_TRUE(down.is_terminal());
  EXPECT_EQ(down.winner(), r::Player::Black);
}

TEST(ConnectX, FullBoardIsADraw){
  r::ConnectXGameState<2, 2, 3> g;
  drops(g, {0, 0, 1});
  EXPECT_FALSE(g.is_terminal());
  drops(g, {1});
  EXPECT_TRUE(g.is_terminal());
  EXPECT_EQ(g.winner(), r::Player::Unknown);
  EXPECT_EQ(g.score(r::Player::Black), 0);
}

TEST(ConnectX, FullColumnIsNotLegal){
  Connect4 g;
  drops(g, {0, 0, 0, 0, 0, 0});
  for (const r::Move& m : g.legal_moves()) EXPECT_NE(m.mpt.c, 0);
  EXPECT_EQ(g.legal_moves().size(), 6u);
  EXPECT_FALSE(g.apply(drop(g, 0)));
}

TEST(ConnectX, RollbackRestoresState){
  Connect4 g;
  drops(g, {3, 3, 2});
  size_t h = g.hash();
  drops(g, {4, 5});
  g.rollback(2);
  EXPECT_EQ(g.hash(), h);
  EXPECT_EQ(g.next_player(), r::Player::White);
  g.rollback(3);
  EXPECT_EQ(g.hash(), Connect4().hash());
}

// regression: moving a board must keep its column heights
TEST(ConnectX, MovedBoardKeepsColumnHeights){
  r::ConnectXBoard<6, 7> b;
  b.place(r::Player::Black, r::Pt(0, 2));
  b.place(r::Player::White, r::Pt(0, 2));
  r::ConnectXBoard<6, 7> moved(std::move(b));
  EXPECT_EQ(moved.tops()[2], 2);
  EXPECT_EQ(moved.tops()[0], 0);
  moved.place(r::Player::Black, r::Pt(0, 2));
  EXPECT_EQ(moved[r::Pt(2, 2)], r::Player::Black);
}

namespace {

template <ubyte R, ubyte C, ubyte X>
void check_termination_against_brute_force(int games, uint64_t seed){
  using G = r::ConnectXGameState<R, C, X>;
  std::mt19937_64 rng(seed);
  for (int game = 0; game < games; ++game){
    G g;
    std::vector<size_t> hashes;
    while (not g.is_terminal()){
      auto moves = g.legal_moves();
      ASSERT_FALSE(moves.empty());
      r::Move m = moves[std::uniform_int_distribution<int>(0, (int)moves.size() - 1)(rng)];
      hashes.push_back(g.hash());
      ASSERT_TRUE(g.apply(r::PlayerMove(g.next_player(), m)));
      r::Player ref = reference_winner<R, C, X>(g.board());
      if (ref != r::Player::Unknown){
        ASSERT_TRUE(g.is_terminal()) << "game " << game << " missed a win\n" << g.board();
        ASSERT_EQ(g.winner(), ref) << "game " << game << "\n" << g.board();
      }
    }
    if (g.winner() != r::Player::Unknown)
      ASSERT_EQ(g.winner(), (reference_winner<R, C, X>(g.board()))) << "game " << game << "\n" << g.board();
    // unwinding one move at a time revisits every earlier position
    while (not hashes.empty()){
      g.rollback(1);
      ASSERT_FALSE(g.is_terminal());
      ASSERT_EQ(g.hash(), hashes.back()) << "game " << game;
      hashes.pop_back();
    }
  }
}

} // namespace

TEST(ConnectX, TerminationMatchesBruteForce){
  check_termination_against_brute_force<6, 7, 4>(3000, 9);
  check_termination_against_brute_force<5, 5, 3>(1000, 10);
  check_termination_against_brute_force<4, 4, 3>(1000, 11);
  check_termination_against_brute_force<3, 3, 3>(1000, 12);
  check_termination_against_brute_force<4, 9, 5>(1000, 13);
}
