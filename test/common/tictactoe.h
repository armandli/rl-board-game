#ifndef RLBG_TEST_TICTACTOE_H
#define RLBG_TEST_TICTACTOE_H

#include <array>
#include <bit>
#include <cstdint>
#include <utility>
#include <vector>

#include <muzero_game.h>

#include <encoder/encoder.h>

namespace rlbg::test {

namespace s = std;

// tic-tac-toe as a MuZeroGame: small enough to know the right move, and full
// of transpositions, which exercises graph search
struct TicTacToe {
  static constexpr int NUM_ACTIONS = 9;
  static constexpr s::array<uint16_t, 8> LINES = {
    0007, 0070, 0700, 0111, 0222, 0444, 0421, 0124,
  };

  int to_play() const { return mPlayer; }
  bool is_terminal() const { return mTerminal; }
  uint16_t own() const { return mOwn; }
  uint16_t opp() const { return mOpp; }
  uint16_t black() const { return mPlayer == 0 ? mOwn : mOpp; }
  uint16_t white() const { return mPlayer == 0 ? mOpp : mOwn; }
  int winner() const { return mWinner; }

  int outcome(int player) const {
    if (mWinner < 0) return 0;
    return mWinner == player ? 1 : -1;
  }
  void legal_actions(ActionList<NUM_ACTIONS>& list) const {
    list.size = 0;
    if (mTerminal) return;
    for (int i = 0; i < 9; ++i)
      if (not (((mOwn | mOpp) >> i) & 1)) list.push_back(i);
  }
  void apply(int a){
    mOwn |= 1 << a;
    if (wins(mOwn)){
      mTerminal = true;
      mWinner = mPlayer;
    } else if ((mOwn | mOpp) == 0777){
      mTerminal = true;
    }
    s::swap(mOwn, mOpp);
    mPlayer ^= 1;
  }
  uint64_t hash() const {
    return (uint64_t)black() | ((uint64_t)white() << 9) | ((uint64_t)mPlayer << 18);
  }

  static bool wins(uint16_t b){
    for (uint16_t l : LINES)
      if ((b & l) == l) return true;
    return false;
  }

protected:
  uint16_t mOwn = 0;
  uint16_t mOpp = 0;
  int8_t   mPlayer = 0;
  bool     mTerminal = false;
  int8_t   mWinner = -1;
};

static_assert(MuZeroGame<TicTacToe>);

// features {empty, own, opponent} per square, the same layout as othello's
// first three features
struct TicTacToeEncoder : Encoder<TicTacToe> {
  TicTacToeEncoder(){
    for (int sym = 0; sym < 8; ++sym){
      s::vector<int> m(9);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c){
          int rr = r, cc = c;
          if (sym >= 4) s::swap(rr, cc);
          for (int i = 0; i < sym % 4; ++i){
            int nr = cc;
            int nc = 2 - rr;
            rr = nr;
            cc = nc;
          }
          m[r * 3 + c] = rr * 3 + cc;
        }
      mMaps.push_back(m);
    }
  }

  int num_tokens() const override { return 9; }
  int num_features() const override { return 3; }
  int num_actions() const override { return 9; }
  int num_token_classes() const override { return 3; }

  void encode(const TicTacToe& g, uint8_t* f) const override {
    for (int i = 0; i < 9; ++i){
      uint8_t o = (g.own() >> i) & 1;
      uint8_t p = (g.opp() >> i) & 1;
      f[i * 3] = 1 - o - p;
      f[i * 3 + 1] = o;
      f[i * 3 + 2] = p;
    }
  }
  void encode_tokens(const TicTacToe& g, uint8_t* t) const override {
    for (int i = 0; i < 9; ++i) t[i] = ((g.own() >> i) & 1) ? 1 : ((g.opp() >> i) & 1) ? 2 : 0;
  }
  void encode_tokens_absolute(const TicTacToe& g, uint8_t* t) const override {
    for (int i = 0; i < 9; ++i) t[i] = ((g.black() >> i) & 1) ? 1 : ((g.white() >> i) & 1) ? 2 : 0;
  }
  int flip_token_class(int c) const override { return c == 0 ? 0 : 3 - c; }
  int num_symmetries() const override { return 8; }
  const s::vector<int>& token_map(int sym) const override { return mMaps[sym]; }
  const s::vector<int>& action_map(int sym) const override { return mMaps[sym]; }

protected:
  s::vector<s::vector<int>> mMaps;
};

}  // namespace rlbg::test

#endif//RLBG_TEST_TICTACTOE_H
