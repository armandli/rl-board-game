#ifndef RLBG_ENCODER_WALL_GO_V1_ENCODER_H
#define RLBG_ENCODER_WALL_GO_V1_ENCODER_H

#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>
#include <vector>

#include <wall_go_v1.h>

#include <encoder/encoder.h>

namespace rlbg {

namespace s = std;

// one token per cell (0..48) and per interior edge (49..132), so token i is
// also action i and pass is the extra action the model expects. Every feature
// is relative to the side to move:
//   0..3   token class: 0 empty edge or cell outside any territory, 1 own
//          territory, 2 opponent territory, 3 wall
//   4, 5   own piece, opponent piece
//   6      active piece: the one selected (Move phase) or just moved (Wall)
//   7      legal action on this token
//   8      cell of a region holding pieces of both players
//   9..12  phase one-hot: place, select, move, wall
//   13..16 own walls left >= 1, 3, 7, 13
//   17..20 opponent walls left >= 1, 3, 7, 13
// Phase and walls left repeat on every token. The game is Markov in this
// state, so no history is needed. Territory classes double as the final
// ownership target. The 8 symmetries of the square map edges along with the
// cells they separate.
template <WallGoV1Rules R = WALL_GO_V1_RULES>
struct WallGoV1Encoder : Encoder<WallGoV1Game<R>> {
  using Game = WallGoV1Game<R>;

  static constexpr int NUM_TOKENS   = Game::NUM_CELLS + Game::NUM_EDGES;
  static constexpr int NUM_CLASSES  = 4;
  static constexpr int NUM_FEATURES = 21;
  static constexpr int WALL_LEVELS[4] = {1, 3, 7, 13};

  WallGoV1Encoder(){
    constexpr int N = Game::SIZE;
    for (int sym = 0; sym < 8; ++sym){
      s::vector<int> tmap(NUM_TOKENS);
      for (int i = 0; i < Game::NUM_CELLS; ++i){
        auto [r, c] = transform(sym, i / N, i % N);
        tmap[i] = r * N + c;
      }
      for (int e = 0; e < Game::NUM_EDGES; ++e){
        auto [a, b] = Game::edge_cells(e);
        tmap[Game::FIRST_EDGE + e] = Game::FIRST_EDGE + edge_between(tmap[a], tmap[b]);
      }
      s::vector<int> amap = tmap;
      amap.push_back(Game::PASS);
      mTokenMaps.push_back(s::move(tmap));
      mActionMaps.push_back(s::move(amap));
    }
  }

  int num_tokens() const override { return NUM_TOKENS; }
  int num_features() const override { return NUM_FEATURES; }
  int num_actions() const override { return Game::NUM_ACTIONS; }
  int num_token_classes() const override { return NUM_CLASSES; }

  void encode(const Game& game, uint8_t* features) const override {
    s::fill(features, features + NUM_TOKENS * NUM_FEATURES, 0);
    int me = game.to_play();
    uint8_t tokens[NUM_TOKENS];
    fill_tokens(game, me, tokens);

    uint64_t contested = 0;
    for (uint64_t rest = (1ULL << Game::NUM_CELLS) - 1; rest;){
      uint64_t reg = game.region(rest & -rest);
      rest &= ~reg;
      if ((reg & game.pieces(0)) and (reg & game.pieces(1))) contested |= reg;
    }
    ActionList<Game::NUM_ACTIONS> legal;
    game.legal_actions(legal);
    bool legal_token[NUM_TOKENS] = {};
    for (int a : legal)
      if (a < NUM_TOKENS) legal_token[a] = true;

    uint8_t shared[NUM_FEATURES] = {};
    if (not game.is_terminal()) shared[9 + (int)game.phase()] = 1;
    for (int k = 0; k < 4; ++k){
      shared[13 + k] = game.walls_left(me) >= WALL_LEVELS[k];
      shared[17 + k] = game.walls_left(me ^ 1) >= WALL_LEVELS[k];
    }

    for (int i = 0; i < NUM_TOKENS; ++i){
      uint8_t* f = features + i * NUM_FEATURES;
      for (int k = 9; k < NUM_FEATURES; ++k) f[k] = shared[k];
      f[tokens[i]] = 1;
      f[7] = legal_token[i];
      if (i < Game::NUM_CELLS){
        f[4] = (game.pieces(me) >> i) & 1;
        f[5] = (game.pieces(me ^ 1) >> i) & 1;
        f[6] = i == game.active_cell();
        f[8] = (contested >> i) & 1;
      }
    }
  }
  void encode_tokens(const Game& game, uint8_t* tokens) const override {
    fill_tokens(game, game.to_play(), tokens);
  }
  void encode_tokens_absolute(const Game& game, uint8_t* tokens) const override {
    fill_tokens(game, 0, tokens);
  }
  int flip_token_class(int token_class) const override {
    return token_class == 1 or token_class == 2 ? 3 - token_class : token_class;
  }

  int num_symmetries() const override { return 8; }
  const s::vector<int>& token_map(int sym) const override { return mTokenMaps[sym]; }
  const s::vector<int>& action_map(int sym) const override { return mActionMaps[sym]; }

  // rotations by 0, 90, 180, 270 degrees, then the same after a transpose
  static s::pair<int,int> transform(int sym, int r, int c){
    if (sym >= 4) s::swap(r, c);
    for (int i = 0; i < sym % 4; ++i){
      int nr = c;
      int nc = Game::SIZE - 1 - r;
      r = nr;
      c = nc;
    }
    return {r, c};
  }
  // the edge separating two neighbouring cells
  static int edge_between(int a, int b){
    if (a > b) s::swap(a, b);
    return b - a == 1 ? Game::east_edge(a) : Game::south_edge(a);
  }

protected:
  // classes seen from player
  static void fill_tokens(const Game& game, int player, uint8_t* tokens){
    uint64_t t[2];
    game.territories(t);
    for (int i = 0; i < Game::NUM_CELLS; ++i)
      tokens[i] = ((t[player] >> i) & 1) ? 1 : ((t[player ^ 1] >> i) & 1) ? 2 : 0;
    for (int e = 0; e < Game::NUM_EDGES; ++e){
      auto [is_east, cell] = Game::edge_cell(e);
      uint64_t walls = is_east ? game.walls_east() : game.walls_south();
      tokens[Game::FIRST_EDGE + e] = ((walls >> cell) & 1) ? 3 : 0;
    }
  }

  s::vector<s::vector<int>> mTokenMaps;
  s::vector<s::vector<int>> mActionMaps;
};

}  // namespace rlbg

#endif//RLBG_ENCODER_WALL_GO_V1_ENCODER_H
