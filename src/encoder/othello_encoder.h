#ifndef RLBG_ENCODER_OTHELLO_ENCODER_H
#define RLBG_ENCODER_OTHELLO_ENCODER_H

#include <cstdint>
#include <vector>

#include <othello_bitboard.h>

#include <encoder/encoder.h>

namespace rlbg {

namespace s = std;

// one token per square with features {empty, own disc, opponent disc, legal
// placement for the side to move}, all relative to the side to move so the
// network always evaluates from the mover's perspective. Othello is Markov
// so no history planes are needed. Token classes are 0 empty, 1 own, 2
// opponent (absolute: 1 black, 2 white). The 8 symmetries of the square
// (dihedral group D4) map pass to itself.
struct OthelloEncoder : Encoder<OthelloBitboard> {
  static constexpr int NUM_FEATURES = 4;

  OthelloEncoder(){
    for (int sym = 0; sym < 8; ++sym){
      s::vector<int> tmap(64);
      for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c){
          auto [tr, tc] = transform(sym, r, c);
          tmap[r * 8 + c] = tr * 8 + tc;
        }
      s::vector<int> amap = tmap;
      amap.push_back(OthelloBitboard::PASS);
      mTokenMaps.push_back(s::move(tmap));
      mActionMaps.push_back(s::move(amap));
    }
  }

  int num_tokens() const override { return OthelloBitboard::NUM_SQUARES; }
  int num_features() const override { return NUM_FEATURES; }
  int num_actions() const override { return OthelloBitboard::NUM_ACTIONS; }
  int num_token_classes() const override { return 3; }

  void encode(const OthelloBitboard& game, uint8_t* features) const override {
    uint64_t own = game.own();
    uint64_t opp = game.opp();
    uint64_t legal = game.is_terminal() ? 0 : game.legal();
    for (int i = 0; i < 64; ++i){
      uint8_t o = (own >> i) & 1;
      uint8_t p = (opp >> i) & 1;
      uint8_t* f = features + i * NUM_FEATURES;
      f[0] = 1 - o - p;
      f[1] = o;
      f[2] = p;
      f[3] = (legal >> i) & 1;
    }
  }
  void encode_tokens(const OthelloBitboard& game, uint8_t* tokens) const override {
    fill_tokens(game.own(), game.opp(), tokens);
  }
  void encode_tokens_absolute(const OthelloBitboard& game, uint8_t* tokens) const override {
    fill_tokens(game.black(), game.white(), tokens);
  }
  int flip_token_class(int token_class) const override {
    return token_class == 0 ? 0 : 3 - token_class;
  }

  int num_symmetries() const override { return 8; }
  const s::vector<int>& token_map(int sym) const override { return mTokenMaps[sym]; }
  const s::vector<int>& action_map(int sym) const override { return mActionMaps[sym]; }

protected:
  static void fill_tokens(uint64_t first, uint64_t second, uint8_t* tokens){
    for (int i = 0; i < 64; ++i)
      tokens[i] = ((first >> i) & 1) ? 1 : ((second >> i) & 1) ? 2 : 0;
  }
  // rotations by 0, 90, 180, 270 degrees, then the same after a transpose
  static s::pair<int,int> transform(int sym, int r, int c){
    if (sym >= 4) s::swap(r, c);
    for (int i = 0; i < sym % 4; ++i){
      int nr = c;
      int nc = 7 - r;
      r = nr;
      c = nc;
    }
    return {r, c};
  }

  s::vector<s::vector<int>> mTokenMaps;
  s::vector<s::vector<int>> mActionMaps;
};

}  // namespace rlbg

#endif//RLBG_ENCODER_OTHELLO_ENCODER_H
