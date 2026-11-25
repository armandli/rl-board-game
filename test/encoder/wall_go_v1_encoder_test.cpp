#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <wall_go_v1.h>

#include <encoder/wall_go_v1_encoder.h>

namespace r = rlbg;

using Game = r::WallGoV1;
using Enc = r::WallGoV1Encoder<>;

namespace {

// random games as their action sequences, cut at random lengths
std::vector<std::vector<int>> random_games(int n, uint64_t seed){
  std::mt19937_64 rng(seed);
  std::vector<std::vector<int>> out;
  for (int i = 0; i < n; ++i){
    Game g;
    std::vector<int> seq;
    r::ActionList<Game::NUM_ACTIONS> legal;
    int stop = std::uniform_int_distribution<int>(0, 200)(rng);
    while (not g.is_terminal() and (int)seq.size() < stop){
      g.legal_actions(legal);
      int a = legal[std::uniform_int_distribution<int>(0, legal.size - 1)(rng)];
      seq.push_back(a);
      g.apply(a);
    }
    out.push_back(seq);
  }
  return out;
}

Game replay(const std::vector<int>& seq, const std::vector<int>& amap){
  Game g;
  for (int a : seq){
    EXPECT_TRUE(g.is_legal(amap[a]));
    g.apply(amap[a]);
  }
  return g;
}

std::vector<int> identity(){
  std::vector<int> v(Game::NUM_ACTIONS);
  for (int i = 0; i < Game::NUM_ACTIONS; ++i) v[i] = i;
  return v;
}

} // namespace

TEST(WallGoV1Encoder, Dimensions){
  Enc enc;
  EXPECT_EQ(enc.num_tokens(), 133);
  EXPECT_EQ(enc.num_actions(), 134);
  // the transformer expects one action per token plus pass
  EXPECT_EQ(enc.num_actions(), enc.num_tokens() + 1);
  EXPECT_EQ(enc.num_features(), 21);
  EXPECT_EQ(enc.num_token_classes(), 4);
  EXPECT_GT(enc.num_features(), enc.num_token_classes());
  EXPECT_EQ(enc.num_symmetries(), 8);
}

TEST(WallGoV1Encoder, SymmetriesArePermutations){
  Enc enc;
  for (int sym = 0; sym < 8; ++sym){
    const std::vector<int>& t = enc.token_map(sym);
    const std::vector<int>& a = enc.action_map(sym);
    EXPECT_EQ(std::set<int>(t.begin(), t.end()).size(), 133u);
    EXPECT_EQ(std::set<int>(a.begin(), a.end()).size(), 134u);
    EXPECT_EQ(a[Game::PASS], Game::PASS);
    for (int i = 0; i < 133; ++i){
      EXPECT_EQ(a[i], t[i]);
      // cells stay cells, edges stay edges
      EXPECT_EQ(i < Game::NUM_CELLS, t[i] < Game::NUM_CELLS);
    }
    if (sym == 0)
      for (int i = 0; i < 133; ++i) EXPECT_EQ(t[i], i);
  }
}

// the rules are invariant under the symmetries: playing the mapped actions
// gives the mapped position, whose encoding is the permuted encoding
TEST(WallGoV1Encoder, EncodingCommutesWithSymmetries){
  Enc enc;
  int T = enc.num_tokens();
  int F = enc.num_features();
  for (const std::vector<int>& seq : random_games(40, 11)){
    Game g = replay(seq, identity());
    std::vector<uint8_t> feat(T * F), tok(T);
    enc.encode(g, feat.data());
    enc.encode_tokens(g, tok.data());
    for (int sym = 1; sym < 8; ++sym){
      Game h = replay(seq, enc.action_map(sym));
      EXPECT_EQ(g.is_terminal(), h.is_terminal());
      EXPECT_EQ(g.outcome(0), h.outcome(0));
      std::vector<uint8_t> hf(T * F), ht(T);
      enc.encode(h, hf.data());
      enc.encode_tokens(h, ht.data());
      const std::vector<int>& map = enc.token_map(sym);
      for (int i = 0; i < T; ++i){
        EXPECT_EQ(tok[i], ht[map[i]]);
        for (int f = 0; f < F; ++f)
          ASSERT_EQ(feat[i * F + f], hf[map[i] * F + f]) << "sym " << sym << " token " << i << " feature " << f;
      }
    }
  }
}

TEST(WallGoV1Encoder, FeaturesDescribeTheBoardFromTheMover){
  Enc enc;
  int T = enc.num_tokens();
  int F = enc.num_features();
  for (const std::vector<int>& seq : random_games(60, 12)){
    Game g = replay(seq, identity());
    std::vector<uint8_t> feat(T * F), tok(T), abs(T);
    enc.encode(g, feat.data());
    enc.encode_tokens(g, tok.data());
    enc.encode_tokens_absolute(g, abs.data());
    int me = g.to_play();

    r::ActionList<Game::NUM_ACTIONS> legal;
    g.legal_actions(legal);
    std::set<int> legal_set(legal.begin(), legal.end());

    for (int i = 0; i < T; ++i){
      const uint8_t* f = feat.data() + i * F;
      // one class, the token class
      EXPECT_EQ(f[0] + f[1] + f[2] + f[3], 1);
      EXPECT_EQ(f[tok[i]], 1);
      EXPECT_EQ(tok[i], me == 0 ? abs[i] : enc.flip_token_class(abs[i]));
      EXPECT_EQ(f[7], legal_set.count(i));
      if (i < Game::NUM_CELLS){
        EXPECT_EQ(f[4], (g.pieces(me) >> i) & 1);
        EXPECT_EQ(f[5], (g.pieces(me ^ 1) >> i) & 1);
        EXPECT_EQ(f[6], i == g.active_cell());
        EXPECT_LT(tok[i], 3);
        if (tok[i] == 1) EXPECT_TRUE((g.territory(me) >> i) & 1);
      } else {
        auto [is_east, cell] = Game::edge_cell(i - Game::FIRST_EDGE);
        bool wall = (((is_east ? g.walls_east() : g.walls_south()) >> cell) & 1);
        EXPECT_EQ(tok[i] == 3, wall);
        EXPECT_EQ(f[4] + f[5] + f[6] + f[8], 0);
      }
      if (not g.is_terminal()) EXPECT_EQ(f[9 + (int)g.phase()], 1);
      EXPECT_EQ(f[13], g.walls_left(me) >= 1);
      EXPECT_EQ(f[17], g.walls_left(me ^ 1) >= 1);
    }
  }
}
