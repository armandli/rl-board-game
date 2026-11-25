#ifndef RLBG_WALL_GO_V1
#define RLBG_WALL_GO_V1

#include <cassert>
#include <cstdint>

#include <array>
#include <bit>
#include <ostream>

#include <muzero_game.h>

namespace rlbg {

namespace s = std;

// rule variants; the sources disagree, see WallGoV1Game
struct WallGoV1Rules {
  int  pieces_per_player = 4;     // 2 player game: 4 (playwallgo.com) or 2 (boardandpieces.com)
  int  max_steps = 2;             // 2: move 0, 1 or 2 steps; 1: exactly one step
  int  walls_per_player = 20;     // 42 each covers all 84 interior edges, i.e. unlimited
  bool wall_beside_piece = true;  // wall must border the moved piece's cell, else any interior edge
  bool corner_start = false;      // 2 pieces each start in opposite corners, no placement phase
};

// rules of playwallgo.com for 2 players, with the wall placed beside the moved
// piece as on boardandpieces.com (playwallgo.com does not say where it goes)
inline constexpr WallGoV1Rules WALL_GO_V1_RULES{};
// the simpler 2 pawn tutorial rules of boardandpieces.com
inline constexpr WallGoV1Rules WALL_GO_V1_TUTORIAL_RULES{2, 1, 42, true, true};

// Wall Go v1, one of several Wall Go rule versions: 2 players on a 7x7
// board, as bitboards with cell index r * 7 + c.
//
// Placement: players alternately put their pieces on empty cells, player 0
// first (skipped with corner_start). Then a turn is: move one piece up to
// max_steps orthogonal steps through empty cells without crossing a wall
// (with 2 steps it may end where it started, which needs a free neighbour),
// then place a wall on a free interior edge. A player who cannot move any
// piece, or has no walls left, cannot complete a turn and passes. The board
// edge needs no walls.
//
// Walls split the board into regions. The game ends when no region holds
// pieces of both players, or when neither player can act. Each player scores
// the cells of the regions holding only their pieces; more cells wins,
// equal is a draw.
//
// A turn is three actions by the same player so the action space stays small:
// select a piece (its cell), pick its destination (a cell), place a wall. The
// actions are the 49 cells (0..48), the 84 interior edges (49..132: 42 edges
// east of cells with c < 6 at 49 + r * 6 + c, then 42 edges south of cells
// with r < 6 at 91 + r * 7 + c) and pass (133).
template <WallGoV1Rules R = WALL_GO_V1_RULES>
struct WallGoV1Game {
  static_assert(R.pieces_per_player >= 1 and R.max_steps >= 1 and R.walls_per_player >= 1);
  static_assert(not R.corner_start or R.pieces_per_player == 2, "corner start places 2 pieces each");

  static constexpr int SIZE        = 7;
  static constexpr int NUM_CELLS   = 49;
  static constexpr int NUM_EDGES   = 84;
  static constexpr int EAST_EDGES  = 42;
  static constexpr int FIRST_EDGE  = NUM_CELLS;
  static constexpr int PASS        = NUM_CELLS + NUM_EDGES;
  static constexpr int NUM_ACTIONS = PASS + 1;

  enum class Phase : uint8_t { Place = 0, Select = 1, Move = 2, Wall = 3 };

  WallGoV1Game(){
    mWalls.fill(R.walls_per_player);
    if (R.corner_start){
      mPieces[0] = bit(0) | bit(NUM_CELLS - 1);
      mPieces[1] = bit(SIZE - 1) | bit(NUM_CELLS - SIZE);
      start_turns();
    }
  }

  int to_play() const { return mPlayer; }
  bool is_terminal() const { return mTerminal; }
  Phase phase() const { return mPhase; }
  uint64_t pieces(int player) const { return mPieces[player]; }
  uint64_t own() const { return mPieces[mPlayer]; }
  uint64_t opp() const { return mPieces[mPlayer ^ 1]; }
  // bit i: wall between cell i and i + 1 / i + 7
  uint64_t walls_east() const { return mWallE; }
  uint64_t walls_south() const { return mWallS; }
  int walls_left(int player) const { return mWalls[player]; }
  // the selected piece in the Move phase, the moved piece in the Wall phase,
  // -1 otherwise
  int active_cell() const { return mActive; }

  // cells of the regions holding only player's pieces
  uint64_t territory(int player) const {
    uint64_t t[2];
    territories(t);
    return t[player];
  }
  int score(int player) const { return s::popcount(territory(player)); }

  int outcome(int player) const {
    uint64_t t[2];
    territories(t);
    int a = s::popcount(t[player]);
    int b = s::popcount(t[player ^ 1]);
    return (a > b) - (b > a);
  }

  void legal_actions(ActionList<NUM_ACTIONS>& list) const {
    list.size = 0;
    if (mTerminal) return;
    switch (mPhase){
    case Phase::Place:
      push_cells(list, BOARD & ~(mPieces[0] | mPieces[1]));
      break;
    case Phase::Select:
      if (mCanAct) push_cells(list, movable(mPlayer));
      else         list.push_back(PASS);
      break;
    case Phase::Move:
      push_cells(list, destinations(mActive));
      break;
    case Phase::Wall:{
      auto [east, south] = legal_walls();
      for (uint64_t m = east; m; m &= m - 1)
        list.push_back(FIRST_EDGE + east_edge(s::countr_zero(m)));
      for (uint64_t m = south; m; m &= m - 1)
        list.push_back(FIRST_EDGE + south_edge(s::countr_zero(m)));
    } break;
    }
  }

  bool is_legal(int action) const {
    if (mTerminal or action < 0 or action >= NUM_ACTIONS) return false;
    switch (mPhase){
    case Phase::Place:
      return action < NUM_CELLS and not ((mPieces[0] | mPieces[1]) & bit(action));
    case Phase::Select:
      if (not mCanAct) return action == PASS;
      return action < NUM_CELLS and (movable(mPlayer) & bit(action));
    case Phase::Move:
      return action < NUM_CELLS and (destinations(mActive) & bit(action));
    case Phase::Wall:{
      if (action < FIRST_EDGE or action == PASS) return false;
      auto [east, south] = legal_walls();
      auto [is_east, cell] = edge_cell(action - FIRST_EDGE);
      return ((is_east ? east : south) >> cell) & 1ULL;
    }
    }
    return false;
  }

  // action must be legal
  void apply(int action){
    assert(is_legal(action));
    switch (mPhase){
    case Phase::Place:
      mPieces[mPlayer] |= bit(action);
      mPlayer ^= 1;
      if (s::popcount(mPieces[1]) == R.pieces_per_player) start_turns();
      break;
    case Phase::Select:
      if (action == PASS){
        mPlayer ^= 1;
        update();
      } else {
        mActive = (int8_t)action;
        mPhase = Phase::Move;
      }
      break;
    case Phase::Move:
      mPieces[mPlayer] = (mPieces[mPlayer] & ~bit(mActive)) | bit(action);
      mActive = (int8_t)action;
      mPhase = Phase::Wall;
      break;
    case Phase::Wall:{
      auto [is_east, cell] = edge_cell(action - FIRST_EDGE);
      (is_east ? mWallE : mWallS) |= bit(cell);
      mWalls[mPlayer] -= 1;
      mActive = -1;
      mPlayer ^= 1;
      mPhase = Phase::Select;
      update();
    } break;
    }
  }

  uint64_t hash() const {
    uint64_t small = (uint64_t)mPlayer | (uint64_t)mPhase << 1 | (uint64_t)(uint8_t)mActive << 3
                   | (uint64_t)mWalls[0] << 11 | (uint64_t)mWalls[1] << 19;
    return mix(mPieces[0]) ^ mix(mPieces[1] ^ 0x9e3779b97f4a7c15ULL) ^ mix(mWallE ^ 0xd1b54a32d192ed03ULL)
         ^ mix(mWallS ^ 0x8cb92ba72f3d8dd7ULL) ^ mix(small ^ 0x5851f42d4c957f2dULL);
  }

  // cells the piece on cell can move to, itself included when it may stay
  uint64_t destinations(int cell) const {
    uint64_t blocked = (mPieces[0] | mPieces[1]) & ~bit(cell);
    uint64_t frontier = bit(cell);
    uint64_t reach = 0;
    for (int k = 0; k < R.max_steps; ++k){
      frontier = step(frontier) & ~blocked;
      reach |= frontier;
    }
    return reach;
  }
  // player's pieces with somewhere to go
  uint64_t movable(int player) const {
    uint64_t out = 0;
    for (uint64_t m = mPieces[player]; m; m &= m - 1){
      int cell = s::countr_zero(m);
      if (destinations(cell)) out |= bit(cell);
    }
    return out;
  }
  // cells reachable from cells in one step, ignoring pieces
  uint64_t step(uint64_t cells) const {
    return ((cells & ~COL6 & ~mWallE) << 1)
         | ((cells & ~COL0 & ~(mWallE << 1)) >> 1)
         | ((cells & ~ROW6 & ~mWallS) << SIZE)
         | ((cells & ~ROW0 & ~(mWallS << SIZE)) >> SIZE);
  }
  // the region of cells connected to seed
  uint64_t region(uint64_t seed) const {
    for (uint64_t next = seed | step(seed); next != seed; next = seed | step(seed))
      seed = next;
    return seed;
  }
  // territories of both players: the regions holding only their pieces
  void territories(uint64_t out[2]) const {
    out[0] = out[1] = 0;
    for (uint64_t rest = BOARD; rest;){
      uint64_t reg = region(rest & -rest);
      rest &= ~reg;
      bool a = reg & mPieces[0];
      bool b = reg & mPieces[1];
      if (a and not b) out[0] |= reg;
      if (b and not a) out[1] |= reg;
    }
  }
  // whether some region holds pieces of both players
  bool contested() const {
    for (uint64_t rest = BOARD; rest;){
      uint64_t reg = region(rest & -rest);
      rest &= ~reg;
      if ((reg & mPieces[0]) and (reg & mPieces[1])) return true;
    }
    return false;
  }

  // edges as walls_east / walls_south bits where a wall may go now
  s::pair<uint64_t, uint64_t> legal_walls() const {
    if (mPhase != Phase::Wall) return {0, 0};
    uint64_t east  = ~COL6 & BOARD & ~mWallE;
    uint64_t south = ~ROW6 & BOARD & ~mWallS;
    if (R.wall_beside_piece){
      uint64_t c = bit(mActive);
      east  &= c | (c >> 1);
      south &= c | (c >> SIZE);
    }
    return {east, south};
  }

  // edge index (0..83) <-> (whether it lies east of the cell, the cell)
  static int east_edge(int cell){ return cell / SIZE * (SIZE - 1) + cell % SIZE; }
  static int south_edge(int cell){ return EAST_EDGES + cell; }
  static s::pair<bool, int> edge_cell(int edge){
    if (edge < EAST_EDGES) return {true, edge / (SIZE - 1) * SIZE + edge % (SIZE - 1)};
    return {false, edge - EAST_EDGES};
  }
  // the two cells an edge separates
  static s::pair<int, int> edge_cells(int edge){
    auto [is_east, cell] = edge_cell(edge);
    return {cell, cell + (is_east ? 1 : SIZE)};
  }

  s::ostream& print(s::ostream& out) const {
    out << "   0 1 2 3 4 5 6\n";
    for (int r = 0; r < SIZE; ++r){
      out << r << "  ";
      for (int c = 0; c < SIZE; ++c){
        int i = r * SIZE + c;
        if (mPieces[0] & bit(i))      out << 'X';
        else if (mPieces[1] & bit(i)) out << 'O';
        else                          out << '.';
        if (c < SIZE - 1) out << ((mWallE & bit(i)) ? '|' : ' ');
      }
      out << '\n';
      if (r < SIZE - 1){
        out << "   ";
        for (int c = 0; c < SIZE; ++c)
          out << ((mWallS & bit(r * SIZE + c)) ? '-' : ' ') << (c < SIZE - 1 ? " " : "");
        out << '\n';
      }
    }
    out << "walls left " << (int)mWalls[0] << " / " << (int)mWalls[1] << ", score " << score(0) << " / " << score(1) << '\n';
    return out;
  }

protected:
  static constexpr uint64_t BOARD = (1ULL << NUM_CELLS) - 1;
  static constexpr uint64_t ROW0  = (1ULL << SIZE) - 1;
  static constexpr uint64_t ROW6  = ROW0 << (NUM_CELLS - SIZE);
  static constexpr uint64_t COL0  = 0x0040810204081ULL;
  static constexpr uint64_t COL6  = COL0 << (SIZE - 1);

  static uint64_t bit(int i){ return 1ULL << i; }
  static uint64_t mix(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
  static void push_cells(ActionList<NUM_ACTIONS>& list, uint64_t cells){
    for (; cells; cells &= cells - 1)
      list.push_back(s::countr_zero(cells));
  }
  bool can_act(int player) const {
    return mWalls[player] > 0 and movable(player) != 0;
  }
  void start_turns(){
    mPlayer = 0;
    mPhase = Phase::Select;
    update();
  }
  // at the start of a turn: ends the game once the players are walled apart
  // or neither can act
  void update(){
    mCanAct = can_act(mPlayer);
    mTerminal = not contested() or (not mCanAct and not can_act(mPlayer ^ 1));
  }

  s::array<uint64_t, 2> mPieces{0, 0};
  uint64_t              mWallE = 0;
  uint64_t              mWallS = 0;
  s::array<int8_t, 2>   mWalls{};
  int8_t                mActive = -1;
  int8_t                mPlayer = 0;
  Phase                 mPhase = Phase::Place;
  bool                  mCanAct = true;
  bool                  mTerminal = false;
};

using WallGoV1 = WallGoV1Game<WALL_GO_V1_RULES>;
using WallGoV1Tutorial = WallGoV1Game<WALL_GO_V1_TUTORIAL_RULES>;

template <WallGoV1Rules R>
s::ostream& operator<<(s::ostream& out, const WallGoV1Game<R>& game){
  return game.print(out);
}

static_assert(MuZeroGame<WallGoV1>);
static_assert(MuZeroGame<WallGoV1Tutorial>);

}  // namespace rlbg

#endif//RLBG_WALL_GO_V1
