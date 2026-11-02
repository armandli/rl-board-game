#include <algorithm>
#include <array>
#include <random>
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

namespace {

// legal placements and flips by walking rays square by square, independent of
// the bitboard engine
template <ubyte SZ>
struct NaiveOthello {
  std::array<r::Player, SZ * SZ> b;

  explicit NaiveOthello(const r::OthelloBoard<SZ>& board){
    for (uint i = 0; i < SZ * SZ; ++i) b[i] = board[r::point<SZ>(i)];
  }
  bool on(int rr, int cc) const { return rr >= 0 and rr < SZ and cc >= 0 and cc < SZ; }
  r::Player at(int rr, int cc) const { return b[rr * SZ + cc]; }
  // squares flipped by p placing at (rr, cc)
  std::vector<int> flips(r::Player p, int rr, int cc) const {
    std::vector<int> out;
    if (at(rr, cc) != r::Player::Unknown) return out;
    for (r::Dr d : r::EIGHT_WAY){
      std::vector<int> run;
      int y = rr + d.r, x = cc + d.c;
      while (on(y, x) and at(y, x) == r::opponent(p)){
        run.push_back(y * SZ + x);
        y += d.r; x += d.c;
      }
      if (not run.empty() and on(y, x) and at(y, x) == p)
        out.insert(out.end(), run.begin(), run.end());
    }
    return out;
  }
  std::vector<int> legal(r::Player p) const {
    std::vector<int> out;
    for (int i = 0; i < SZ * SZ; ++i)
      if (not flips(p, i / SZ, i % SZ).empty()) out.push_back(i);
    return out;
  }
};

template <ubyte SZ>
void check_against_naive(int games, uint64_t seed){
  using G = r::OthelloGameState<SZ>;
  std::mt19937_64 rng(seed);
  for (int game = 0; game < games; ++game){
    G g;
    std::vector<size_t> hashes;
    while (not g.is_terminal()){
      NaiveOthello<SZ> ref(g.board());
      std::vector<int> got;
      for (const r::Move& m : g.legal_moves())
        if (m.mty == r::M::Play) got.push_back(r::index<SZ>(m.mpt));
      std::sort(got.begin(), got.end());
      ASSERT_EQ(got, ref.legal(g.next_player())) << "game " << game;

      hashes.push_back(g.hash());
      if (got.empty()){
        ASSERT_TRUE(g.apply(r::PlayerMove(g.next_player(), r::Move(r::M::Pass))));
        continue;
      }
      int a = got[std::uniform_int_distribution<size_t>(0, got.size() - 1)(rng)];
      r::Player p = g.next_player();
      std::vector<int> f = ref.flips(p, a / SZ, a % SZ);
      ASSERT_TRUE(g.apply(r::PlayerMove(p, r::Move(r::M::Play, r::point<SZ>(a)))));
      for (int i = 0; i < SZ * SZ; ++i){
        r::Player want = ref.b[i];
        if (i == a or std::find(f.begin(), f.end(), i) != f.end()) want = p;
        ASSERT_EQ(g.board()[r::point<SZ>(i)], want) << "game " << game << " square " << i;
      }
    }
    // unwinding one move at a time revisits every earlier position
    while (not hashes.empty()){
      g.rollback(1);
      ASSERT_FALSE(g.is_terminal());
      ASSERT_EQ(g.hash(), hashes.back()) << "game " << game;
      hashes.pop_back();
    }
    ASSERT_EQ(g.hash(), G().hash());
  }
}

} // namespace

TEST(Othello, MatchesNaiveRules8x8){ check_against_naive<8>(500, 11); }
TEST(Othello, MatchesNaiveRules6x6){ check_against_naive<6>(500, 12); }
TEST(Othello, MatchesNaiveRules4x4){ check_against_naive<4>(500, 13); }
