#ifndef RLBG_EVALUATION_OTHELLO_ALPHA_BETA_AGENT_H
#define RLBG_EVALUATION_OTHELLO_ALPHA_BETA_AGENT_H

#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <random>
#include <string>
#include <vector>

#include <othello_bitboard.h>

#include <evaluation/agent.h>

namespace rlbg {

namespace s = std;

// a classic othello engine as a fixed evaluation yardstick that knows
// nothing learned: negamax alpha-beta to a fixed depth over a positional
// square table plus mobility and corner occupancy, searched exactly to the
// end of the game once few empty squares remain. Ties between equally scored
// moves are broken at random so matches vary.
struct OthelloAlphaBetaAgent : Agent<OthelloBitboard> {
  OthelloAlphaBetaAgent(int depth, uint64_t seed, int exact_empties = 8)
  : mDepth(depth), mExact(exact_empties), mRng(seed) {}

  s::string name() const override { return s::format("alphabeta_{}", mDepth); }

  void select_actions(const s::vector<const OthelloBitboard*>& games, s::vector<int>& actions) override {
    actions.resize(games.size());
    for (size_t i = 0; i < games.size(); ++i) actions[i] = search(*games[i]);
  }

  // best move for the side to move
  int search(const OthelloBitboard& g){
    uint64_t own = g.own(), opp = g.opp();
    uint64_t moves = OthelloBitboard::legal_moves(own, opp);
    if (moves == 0) return OthelloBitboard::PASS;
    int empties = 64 - s::popcount(own | opp);
    int depth = empties <= mExact ? empties + 2 : mDepth;
    s::vector<int> order;
    for (uint64_t m = moves; m; m &= m - 1) order.push_back(s::countr_zero(m));
    s::shuffle(order.begin(), order.end(), mRng);
    int best = order[0];
    int alpha = -INF;
    for (int sq : order){
      uint64_t f = OthelloBitboard::flips(own, opp, sq);
      int v = -negamax(opp & ~f, own | f | (1ULL << sq), depth - 1, -INF, -alpha, false);
      if (v > alpha){
        alpha = v;
        best = sq;
      }
    }
    return best;
  }

protected:
  static constexpr int INF = 1 << 28;
  static constexpr int WIN = 1 << 20;
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
  static constexpr uint64_t CORNERS = 0x8100000000000081ULL;

  static int final_score(uint64_t own, uint64_t opp){
    int d = s::popcount(own) - s::popcount(opp);
    return d > 0 ? WIN + d : d < 0 ? -WIN + d : 0;
  }

  static int evaluate(uint64_t own, uint64_t opp, uint64_t own_moves){
    int score = 0;
    for (uint64_t b = own; b; b &= b - 1) score += WEIGHTS[s::countr_zero(b)];
    for (uint64_t b = opp; b; b &= b - 1) score -= WEIGHTS[s::countr_zero(b)];
    int mobility = s::popcount(own_moves) - s::popcount(OthelloBitboard::legal_moves(opp, own));
    int corners = s::popcount(own & CORNERS) - s::popcount(opp & CORNERS);
    return score + 10 * mobility + 30 * corners;
  }

  // value for the side to move (own)
  int negamax(uint64_t own, uint64_t opp, int depth, int alpha, int beta, bool passed){
    uint64_t moves = OthelloBitboard::legal_moves(own, opp);
    if (moves == 0){
      if (passed or OthelloBitboard::legal_moves(opp, own) == 0) return final_score(own, opp);
      return -negamax(opp, own, depth, -beta, -alpha, true);
    }
    if (depth <= 0) return evaluate(own, opp, moves);
    // corners first improves pruning
    int best = -INF;
    for (uint64_t group : {moves & CORNERS, moves & ~CORNERS})
      for (uint64_t m = group; m; m &= m - 1){
        int sq = s::countr_zero(m);
        uint64_t f = OthelloBitboard::flips(own, opp, sq);
        int v = -negamax(opp & ~f, own | f | (1ULL << sq), depth - 1, -beta, -alpha, false);
        best = s::max(best, v);
        alpha = s::max(alpha, v);
        if (alpha >= beta) return best;
      }
    return best;
  }

  int           mDepth;
  int           mExact;
  s::mt19937_64 mRng;
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_OTHELLO_ALPHA_BETA_AGENT_H
