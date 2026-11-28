#include <algorithm>
#include <bit>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <wall_go_v1.h>

namespace r = rlbg;

using G = r::WallGoV1;
using Phase = G::Phase;

namespace {

// one piece each, so a few turns can wall the players apart
constexpr r::WallGoV1Rules ONE_PIECE{1, 2, 20, true, false};
// one wall each
constexpr r::WallGoV1Rules ONE_WALL{1, 2, 1, true, false};
// walls go on any free interior edge
constexpr r::WallGoV1Rules ANY_WALL{4, 2, 20, false, false};

constexpr int cell(int r, int c){ return r * 7 + c; }
constexpr int east(int r, int c){ return G::FIRST_EDGE + r * 6 + c; }
constexpr int south(int r, int c){ return G::FIRST_EDGE + 42 + r * 7 + c; }

template <typename Game>
std::vector<int> actions(const Game& g){
  r::ActionList<Game::NUM_ACTIONS> list;
  g.legal_actions(list);
  std::vector<int> out(list.begin(), list.end());
  std::sort(out.begin(), out.end());
  return out;
}

template <typename Game>
void play(Game& g, const std::vector<int>& seq){
  for (int a : seq){
    ASSERT_TRUE(g.is_legal(a)) << "action " << a << "\n" << g;
    g.apply(a);
  }
}

std::vector<int> cells_of(uint64_t bits){
  std::vector<int> out;
  for (; bits; bits &= bits - 1) out.push_back(std::countr_zero(bits));
  return out;
}

// places player 0 on p0 and player 1 on p1 in turn
template <typename Game>
void place(Game& g, const std::vector<int>& p0, const std::vector<int>& p1){
  for (size_t i = 0; i < p0.size(); ++i){
    play(g, {p0[i]});
    play(g, {p1[i]});
  }
}

// random games checking the invariants every step; returns the number of
// actions of the longest game
template <typename Game>
int random_games(int games, uint64_t seed, int max_actions){
  std::mt19937_64 rng(seed);
  int longest = 0;
  for (int n = 0; n < games; ++n){
    Game g;
    int len = 0;
    while (not g.is_terminal()){
      std::vector<int> legal = actions(g);
      EXPECT_FALSE(legal.empty());
      // the list agrees with is_legal on every action
      int count = 0;
      for (int a = 0; a < Game::NUM_ACTIONS; ++a) count += g.is_legal(a);
      EXPECT_EQ(count, (int)legal.size());
      int a = legal[std::uniform_int_distribution<int>(0, (int)legal.size() - 1)(rng)];
      int mover = g.to_play();
      auto phase = g.phase();
      int walls = g.walls_left(0) + g.walls_left(1);
      int pieces = std::popcount(g.pieces(0) | g.pieces(1));
      g.apply(a);
      EXPECT_EQ(std::popcount(g.pieces(0) | g.pieces(1)), pieces + (phase == Game::Phase::Place));
      ++len;
      // the selecting player also moves and places the wall
      if ((phase == Game::Phase::Select and a != Game::PASS) or phase == Game::Phase::Move)
        EXPECT_EQ(g.to_play(), mover);
      EXPECT_EQ(g.walls_left(0) + g.walls_left(1), walls - (phase == Game::Phase::Wall));
      EXPECT_EQ(std::popcount(g.walls_east()) + std::popcount(g.walls_south()),
                2 * Game().walls_left(0) - g.walls_left(0) - g.walls_left(1));
      if (len > max_actions){
        ADD_FAILURE() << "game did not end\n" << g;
        return len;
      }
    }
    EXPECT_TRUE(actions(g).empty());
    EXPECT_EQ(g.outcome(0), -g.outcome(1));
    EXPECT_EQ(g.outcome(0), (g.score(0) > g.score(1)) - (g.score(1) > g.score(0)));
    // pieces never share a cell and every piece is on the board
    EXPECT_EQ(g.pieces(0) & g.pieces(1), 0u);
    EXPECT_EQ((g.pieces(0) | g.pieces(1)) >> 49, 0u);
    longest = std::max(longest, len);
  }
  return longest;
}

} // namespace

TEST(WallGoV1, PlacementAlternatesThenPlayerZeroMoves){
  G g;
  EXPECT_EQ(g.phase(), Phase::Place);
  EXPECT_EQ(g.to_play(), 0);
  EXPECT_EQ(actions(g).size(), 49u);
  EXPECT_EQ(g.walls_left(0), 20);
  EXPECT_EQ(g.walls_left(1), 20);

  std::vector<int> p0{cell(0, 0), cell(2, 2), cell(4, 4), cell(6, 6)};
  std::vector<int> p1{cell(0, 6), cell(2, 4), cell(4, 2), cell(6, 0)};
  for (int i = 0; i < 4; ++i){
    EXPECT_EQ(g.to_play(), 0);
    play(g, {p0[i]});
    EXPECT_FALSE(g.is_legal(p0[i])) << "occupied";
    EXPECT_EQ(g.to_play(), 1);
    if (i < 3) EXPECT_EQ(g.phase(), Phase::Place);
    play(g, {p1[i]});
  }
  EXPECT_EQ(g.phase(), Phase::Select);
  EXPECT_EQ(g.to_play(), 0);
  EXPECT_FALSE(g.is_terminal());
  std::vector<int> sorted = p0;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(actions(g), sorted);
}

TEST(WallGoV1, TutorialStartsInTheCorners){
  r::WallGoV1Tutorial g;
  EXPECT_EQ(g.phase(), r::WallGoV1Tutorial::Phase::Select);
  EXPECT_EQ(cells_of(g.pieces(0)), (std::vector<int>{cell(0, 0), cell(6, 6)}));
  EXPECT_EQ(cells_of(g.pieces(1)), (std::vector<int>{cell(0, 6), cell(6, 0)}));
  play(g, {cell(0, 0)});
  // exactly one step, no staying
  EXPECT_EQ(actions(g), (std::vector<int>{cell(0, 1), cell(1, 0)}));
}

TEST(WallGoV1, MovesUpToTwoStepsAndMayStay){
  G g;
  place(g, {cell(3, 3), cell(0, 0), cell(0, 2), cell(0, 4)}, {cell(6, 0), cell(6, 2), cell(6, 4), cell(6, 6)});
  play(g, {cell(3, 3)});
  EXPECT_EQ(g.phase(), Phase::Move);
  EXPECT_EQ(g.active_cell(), cell(3, 3));
  // every cell within 2 steps, the start included
  std::vector<int> expect;
  for (int r = 0; r < 7; ++r)
    for (int c = 0; c < 7; ++c)
      if (std::abs(r - 3) + std::abs(c - 3) <= 2) expect.push_back(cell(r, c));
  EXPECT_EQ(actions(g), expect);
}

TEST(WallGoV1, PiecesBlockTheWay){
  G g;
  place(g, {cell(3, 3), cell(0, 0), cell(0, 2), cell(0, 4)}, {cell(3, 4), cell(6, 2), cell(6, 4), cell(6, 6)});
  play(g, {cell(3, 3)});
  std::vector<int> dest = actions(g);
  EXPECT_EQ(dest.size(), 11u);
  EXPECT_FALSE(g.is_legal(cell(3, 4))) << "occupied";
  EXPECT_FALSE(g.is_legal(cell(3, 5))) << "only reachable through a piece";
  EXPECT_TRUE(g.is_legal(cell(2, 4)));
  EXPECT_TRUE(g.is_legal(cell(4, 4)));
}

TEST(WallGoV1, WallGoesBesideTheMovedPiece){
  G g;
  place(g, {cell(3, 3), cell(0, 0), cell(0, 2), cell(0, 4)}, {cell(6, 0), cell(6, 2), cell(6, 4), cell(6, 6)});
  play(g, {cell(3, 3), cell(2, 3)});
  EXPECT_EQ(g.phase(), Phase::Wall);
  EXPECT_EQ(g.active_cell(), cell(2, 3));
  EXPECT_EQ(g.to_play(), 0);
  EXPECT_EQ(actions(g), (std::vector<int>{east(2, 2), east(2, 3), south(1, 3), south(2, 3)}));
  play(g, {east(2, 3)});
  EXPECT_EQ(g.to_play(), 1);
  EXPECT_EQ(g.phase(), Phase::Select);
  EXPECT_EQ(g.walls_left(0), 19);

  // player 1 moves; the corner piece has only two interior edges
  play(g, {cell(6, 6), cell(6, 6)});
  EXPECT_EQ(actions(g), (std::vector<int>{east(6, 5), south(5, 6)}));
  play(g, {east(6, 5)});

  // the wall now blocks (2,3) -> (2,4) and anything beyond it
  play(g, {cell(2, 3)});
  EXPECT_FALSE(g.is_legal(cell(2, 4)));
  EXPECT_TRUE(g.is_legal(cell(1, 4)));  // around through (1,3)
  play(g, {cell(2, 3)});
  EXPECT_FALSE(g.is_legal(east(2, 3))) << "walls do not overlap";
}

TEST(WallGoV1, AnyInteriorEdgeVariant){
  r::WallGoV1Game<ANY_WALL> g;
  place(g, {cell(3, 3), cell(0, 0), cell(0, 2), cell(0, 4)}, {cell(6, 0), cell(6, 2), cell(6, 4), cell(6, 6)});
  play(g, {cell(3, 3), cell(3, 3)});
  EXPECT_EQ(actions(g).size(), 84u);
  play(g, {east(0, 0)});
  play(g, {cell(6, 0), cell(5, 0)});
  EXPECT_EQ(actions(g).size(), 83u);
}

TEST(WallGoV1, WallingOffEndsTheGameAndScoresTerritory){
  r::WallGoV1Game<ONE_PIECE> g;
  play(g, {cell(0, 0), cell(6, 6)});
  play(g, {cell(0, 0), cell(0, 0), east(0, 0)});
  EXPECT_FALSE(g.is_terminal());
  EXPECT_EQ(g.score(0), 0);
  play(g, {cell(6, 6), cell(6, 6), east(6, 5)});
  play(g, {cell(0, 0), cell(0, 0), south(0, 0)});
  EXPECT_TRUE(g.is_terminal());
  EXPECT_EQ(g.territory(0), 1ULL);
  EXPECT_EQ(g.score(0), 1);
  EXPECT_EQ(g.score(1), 48);
  EXPECT_EQ(g.outcome(0), -1);
  EXPECT_EQ(g.outcome(1), 1);
  EXPECT_TRUE(actions(g).empty());
}

TEST(WallGoV1, StuckPlayerPasses){
  r::WallGoV1Game<ONE_PIECE> g;
  using P = r::WallGoV1Game<ONE_PIECE>;
  play(g, {cell(0, 0), cell(0, 1)});
  play(g, {cell(0, 0), cell(0, 0), south(0, 0)});
  play(g, {cell(0, 1), cell(0, 1), south(0, 1)});
  // (0,0) is walled below and blocked by the piece on (0,1), but still
  // shares a region with it
  EXPECT_FALSE(g.is_terminal());
  EXPECT_EQ(g.to_play(), 0);
  EXPECT_EQ(actions(g), (std::vector<int>{P::PASS}));
  play(g, {P::PASS});
  EXPECT_EQ(g.to_play(), 1);
  EXPECT_EQ(g.phase(), P::Phase::Select);
  EXPECT_EQ(g.walls_left(0), 19) << "a pass places no wall";
}

TEST(WallGoV1, EndsWhenNoWallsAreLeft){
  r::WallGoV1Game<ONE_WALL> g;
  play(g, {cell(3, 3), cell(5, 5)});
  play(g, {cell(3, 3), cell(3, 3), east(3, 3)});
  EXPECT_FALSE(g.is_terminal());
  play(g, {cell(5, 5), cell(5, 5), east(5, 5)});
  EXPECT_TRUE(g.is_terminal());
  // both pieces share the one region, so nobody owns anything
  EXPECT_EQ(g.score(0), 0);
  EXPECT_EQ(g.score(1), 0);
  EXPECT_EQ(g.outcome(0), 0);
}

TEST(WallGoV1, HashMatchesTranspositions){
  G a, b;
  place(a, {cell(0, 0), cell(1, 1), cell(2, 2), cell(3, 3)}, {cell(6, 6), cell(5, 5), cell(4, 4), cell(3, 4)});
  place(b, {cell(2, 2), cell(3, 3), cell(0, 0), cell(1, 1)}, {cell(4, 4), cell(3, 4), cell(6, 6), cell(5, 5)});
  EXPECT_EQ(a.hash(), b.hash());
  play(a, {cell(0, 0)});
  EXPECT_NE(a.hash(), b.hash()) << "the selected piece is part of the state";
  play(b, {cell(1, 1)});
  EXPECT_NE(a.hash(), b.hash());
}

TEST(WallGoV1, HashSeparatesRandomPositions){
  std::mt19937_64 rng(5);
  std::set<uint64_t> hashes;
  int positions = 0;
  for (int n = 0; n < 50; ++n){
    G g;
    while (not g.is_terminal()){
      hashes.insert(g.hash());
      ++positions;
      std::vector<int> legal = actions(g);
      g.apply(legal[std::uniform_int_distribution<int>(0, (int)legal.size() - 1)(rng)]);
    }
  }
  // distinct games rarely meet, so nearly every position hashes differently
  EXPECT_GT(hashes.size(), positions * 9 / 10);
}

TEST(WallGoV1, RandomGamesKeepTheRulesAndEnd){
  // 8 placements, at most 40 turns of 3 actions and a pass per wall
  int longest = random_games<G>(300, 1, 8 + 40 * 4 + 1);
  EXPECT_GT(longest, 8);
  random_games<r::WallGoV1Tutorial>(300, 2, 84 * 4 + 1);
  random_games<r::WallGoV1Game<ANY_WALL>>(100, 3, 8 + 40 * 4 + 1);
  random_games<r::WallGoV1Game<ONE_PIECE>>(300, 4, 2 + 40 * 4 + 1);
}
