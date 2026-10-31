#ifndef RLBG_TEST_TRUE_MODEL_H
#define RLBG_TEST_TRUE_MODEL_H

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>

#include <othello_bitboard.h>

#include <model/model_inference.h>
#include <search/inference.h>

#include <common/tictactoe.h>

namespace rlbg::test {

namespace s = std;

// a perfect MuZero model of a board game: the latent is the board itself
// (own then opponent discs as 0/1 halves), dynamics apply the real rules and
// the policy is uniform. Searching it checks search conventions independent
// of learning. Rules provides TOKENS, FEATURES, ACTIONS, apply() and value();
// the encoder's features 1 and 2 must be own and opponent discs.
template <typename Rules>
struct TrueModel : UpdatableInference {
  int observation_size() const override { return Rules::TOKENS * Rules::FEATURES; }
  int latent_size() const override { return 2 * Rules::TOKENS; }
  int num_actions() const override { return Rules::ACTIONS; }
  void set_weights(const nn::Params&) override { weight_updates += 1; }

  void infer(const InferenceRequests& req, InferenceResults& res) override {
    constexpr int T = Rules::TOKENS;
    int ni = req.num_initial();
    int n = req.size();
    calls += 1;
    positions += n;
    res.num_initial = ni;
    res.latent_size = 2 * T;
    res.num_actions = Rules::ACTIONS;
    res.latents.assign((size_t)n * 2 * T, latent_t(0));
    res.policy_logits.assign((size_t)n * Rules::ACTIONS, 0.f);
    res.values.assign(n, 0.f);
    res.rewards.assign(n, 0.f);
    for (int i = 0; i < n; ++i){
      uint64_t own = 0, opp = 0;
      if (i < ni){
        const uint8_t* f = req.features.data() + (size_t)i * observation_size();
        for (int sq = 0; sq < T; ++sq){
          if (f[sq * Rules::FEATURES + 1]) own |= 1ULL << sq;
          if (f[sq * Rules::FEATURES + 2]) opp |= 1ULL << sq;
        }
      } else {
        const latent_t* l = req.latents.data() + (size_t)(i - ni) * 2 * T;
        for (int sq = 0; sq < T; ++sq){
          if ((float)l[sq] > 0.5f)     own |= 1ULL << sq;
          if ((float)l[T + sq] > 0.5f) opp |= 1ULL << sq;
        }
        Rules::apply(own, opp, req.actions[i - ni]);
        s::swap(own, opp);
      }
      latent_t* out = res.latents.data() + (size_t)i * 2 * T;
      for (int sq = 0; sq < T; ++sq){
        out[sq] = latent_t((own >> sq) & 1 ? 1.f : 0.f);
        out[T + sq] = latent_t((opp >> sq) & 1 ? 1.f : 0.f);
      }
      res.values[i] = Rules::value(own, opp);
    }
  }

  s::atomic<int> calls{0};
  s::atomic<int> positions{0};
  s::atomic<int> weight_updates{0};
};

// othello with a classic positional plus mobility evaluation
struct OthelloRules {
  static constexpr int TOKENS = 64;
  static constexpr int FEATURES = 4;
  static constexpr int ACTIONS = 65;
  static constexpr int WEIGHTS[64] = {
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
  };

  static void apply(uint64_t& own, uint64_t& opp, int a){
    if (a == OthelloBitboard::PASS) return;
    uint64_t f = OthelloBitboard::flips(own, opp, a);
    own |= f | (1ULL << a);
    opp &= ~f;
  }
  static float value(uint64_t own, uint64_t opp){
    int score = 0;
    for (int sq = 0; sq < 64; ++sq)
      score += WEIGHTS[sq] * ((int)((own >> sq) & 1) - (int)((opp >> sq) & 1));
    int mobility = s::popcount(OthelloBitboard::legal_moves(own, opp)) -
                   s::popcount(OthelloBitboard::legal_moves(opp, own));
    return s::tanh((score + 15 * mobility) / 100.f);
  }
};

// tic-tac-toe valued 0 everywhere: the search learns only from terminal
// rewards, which the real rules supply
struct TicTacToeRules {
  static constexpr int TOKENS = 9;
  static constexpr int FEATURES = 3;
  static constexpr int ACTIONS = 9;

  static void apply(uint64_t& own, uint64_t&, int a){ own |= 1ULL << a; }
  static float value(uint64_t, uint64_t){ return 0.f; }
};

using OthelloTrueModel = TrueModel<OthelloRules>;
using TicTacToeTrueModel = TrueModel<TicTacToeRules>;

}  // namespace rlbg::test

#endif//RLBG_TEST_TRUE_MODEL_H
