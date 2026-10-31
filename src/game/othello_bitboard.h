#ifndef RLBG_OTHELLO_BITBOARD
#define RLBG_OTHELLO_BITBOARD

#include <bit>
#include <cstdint>
#include <ostream>

#include <muzero_game.h>

namespace rlbg {

namespace s = std;

// 8x8 othello on bitboards for search and self-play, where a state is copied
// at every simulation step. Square index is row * 8 + col, action 64 is pass,
// which is legal only when the side to move has no placement (standard rules,
// unlike OthelloGameState which allows passing at any time). The game ends
// when neither side can place.
struct OthelloBitboard {
  static constexpr int SIZE        = 8;
  static constexpr int NUM_SQUARES = 64;
  static constexpr int PASS        = 64;
  static constexpr int NUM_ACTIONS = 65;

  OthelloBitboard()
  : mOwn((1ULL << 28) | (1ULL << 35)), mOpp((1ULL << 27) | (1ULL << 36)),
    mLegal(legal_moves(mOwn, mOpp)), mPlayer(0), mTerminal(false)
  {}

  int to_play() const { return mPlayer; }
  bool is_terminal() const { return mTerminal; }
  uint64_t own() const { return mOwn; }
  uint64_t opp() const { return mOpp; }
  uint64_t black() const { return mPlayer == 0 ? mOwn : mOpp; }
  uint64_t white() const { return mPlayer == 0 ? mOpp : mOwn; }
  // placements for the side to move
  uint64_t legal() const { return mLegal; }

  int outcome(int player) const {
    int b = s::popcount(black());
    int w = s::popcount(white());
    int black_result = (b > w) - (w > b);
    return player == 0 ? black_result : -black_result;
  }

  void legal_actions(ActionList<NUM_ACTIONS>& list) const {
    list.size = 0;
    if (mTerminal) return;
    if (mLegal == 0){
      list.push_back(PASS);
      return;
    }
    for (uint64_t m = mLegal; m; m &= m - 1)
      list.push_back(s::countr_zero(m));
  }

  bool is_legal(int action) const {
    if (mTerminal) return false;
    if (action == PASS) return mLegal == 0;
    return action >= 0 and action < NUM_SQUARES and (mLegal >> action) & 1ULL;
  }

  void apply(int action){
    if (action != PASS){
      uint64_t f = flips(mOwn, mOpp, action);
      mOwn |= f | (1ULL << action);
      mOpp &= ~f;
    }
    s::swap(mOwn, mOpp);
    mPlayer ^= 1;
    mLegal = legal_moves(mOwn, mOpp);
    mTerminal = mLegal == 0 and legal_moves(mOpp, mOwn) == 0;
  }

  uint64_t hash() const {
    return mix(black()) ^ mix(white() ^ 0x9e3779b97f4a7c15ULL) ^ (mPlayer ? 0xd1b54a32d192ed03ULL : 0ULL);
  }

  // placements for player p against opponent o, Kogge-Stone style flood fill
  // along the eight directions
  static uint64_t legal_moves(uint64_t p, uint64_t o){
    uint64_t empty = ~(p | o);
    uint64_t moves = 0;
    for (int i = 0; i < 4; ++i){
      uint64_t mo = o & MASKS[i];
      moves |= sweep(p, mo, DIRS[i]) & empty;
      moves |= sweep(p, mo, -DIRS[i]) & empty;
    }
    return moves;
  }

  // discs of o flipped by p placing on sq
  static uint64_t flips(uint64_t p, uint64_t o, int sq){
    uint64_t f = 0;
    uint64_t m = 1ULL << sq;
    for (int i = 0; i < 4; ++i){
      uint64_t mo = o & MASKS[i];
      for (int d : {DIRS[i], -DIRS[i]}){
        uint64_t x = 0;
        uint64_t t = shift(m, d);
        while (t & mo){
          x |= t;
          t = shift(t, d);
        }
        if (t & p) f |= x;
      }
    }
    return f;
  }

  s::ostream& print(s::ostream& out) const {
    uint64_t b = black();
    uint64_t w = white();
    out << "  01234567\n";
    for (int r = 0; r < SIZE; ++r){
      out << r << ' ';
      for (int c = 0; c < SIZE; ++c){
        int i = r * SIZE + c;
        if ((b >> i) & 1ULL)                          out << 'X';
        else if ((w >> i) & 1ULL)                     out << 'O';
        else if (not mTerminal and (mLegal >> i) & 1) out << '+';
        else                                          out << '.';
      }
      out << '\n';
    }
    return out;
  }

protected:
  // east, south, south-east, south-west; the 0x7e mask keeps runs that move
  // across columns from wrapping around the board edge
  static constexpr int      DIRS[4]  = {1, 8, 9, 7};
  static constexpr uint64_t MASKS[4] = {
    0x7e7e7e7e7e7e7e7eULL, 0xffffffffffffffffULL, 0x7e7e7e7e7e7e7e7eULL, 0x7e7e7e7e7e7e7e7eULL,
  };

  static uint64_t shift(uint64_t b, int d){
    return d > 0 ? b << d : b >> -d;
  }
  // squares one step beyond each run of o that starts next to p
  static uint64_t sweep(uint64_t p, uint64_t mo, int d){
    uint64_t x = shift(p, d) & mo;
    for (int i = 0; i < 5; ++i)
      x |= shift(x, d) & mo;
    return shift(x, d);
  }
  static uint64_t mix(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  uint64_t mOwn;
  uint64_t mOpp;
  uint64_t mLegal;
  int8_t   mPlayer;
  bool     mTerminal;
};

inline s::ostream& operator<<(s::ostream& out, const OthelloBitboard& game){
  return game.print(out);
}

static_assert(MuZeroGame<OthelloBitboard>);

}  // namespace rlbg

#endif//RLBG_OTHELLO_BITBOARD
