#include <bit>
#include <random>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <evaluation/agents.h>
#include <evaluation/arena.h>
#include <evaluation/othello_alpha_beta_agent.h>

namespace r = rlbg;

using Game = r::OthelloBitboard;

TEST(OthelloAlphaBetaAgent, PlaysLegalMovesAndPasses){
  r::OthelloAlphaBetaAgent agent(2, 1);
  std::mt19937_64 rng(1);
  r::ActionList<Game::NUM_ACTIONS> legal;
  for (int game = 0; game < 5; ++game){
    Game g;
    while (not g.is_terminal()){
      int a = agent.search(g);
      ASSERT_TRUE(g.is_legal(a)) << a;
      g.legal_actions(legal);
      // alternate with random moves for varied positions
      g.apply(rng() % 2 ? a : legal[std::uniform_int_distribution<int>(0, legal.size - 1)(rng)]);
    }
  }
}

namespace {

// exposes the search internals and adds plain minimax to check pruning
struct OpenAlphaBeta : r::OthelloAlphaBetaAgent {
  using r::OthelloAlphaBetaAgent::OthelloAlphaBetaAgent;

  int pruned(uint64_t own, uint64_t opp, int depth){ return negamax(own, opp, depth, -INF, INF, false); }

  int minimax(uint64_t own, uint64_t opp, int depth, bool passed){
    uint64_t moves = Game::legal_moves(own, opp);
    if (moves == 0){
      if (passed or Game::legal_moves(opp, own) == 0) return final_score(own, opp);
      return -minimax(opp, own, depth, true);
    }
    if (depth <= 0) return evaluate(own, opp, moves);
    int best = -INF;
    for (uint64_t m = moves; m; m &= m - 1){
      int sq = std::countr_zero(m);
      uint64_t f = Game::flips(own, opp, sq);
      best = std::max(best, -minimax(opp & ~f, own | f | (1ULL << sq), depth - 1, false));
    }
    return best;
  }
};

} // namespace

// pruning and move ordering never change the value of a position
TEST(OthelloAlphaBetaAgent, PrunedSearchMatchesMinimax){
  OpenAlphaBeta agent(3, 1);
  std::mt19937_64 rng(2);
  r::ActionList<Game::NUM_ACTIONS> legal;
  for (int i = 0; i < 40; ++i){
    Game g;
    int moves = std::uniform_int_distribution<int>(0, 55)(rng);
    for (int k = 0; k < moves and not g.is_terminal(); ++k){
      g.legal_actions(legal);
      g.apply(legal[std::uniform_int_distribution<int>(0, legal.size - 1)(rng)]);
    }
    if (g.is_terminal()) continue;
    for (int depth : {1, 2, 4})
      EXPECT_EQ(agent.pruned(g.own(), g.opp(), depth), agent.minimax(g.own(), g.opp(), depth, false))
        << "position " << i << " depth " << depth;
  }
}

TEST(OthelloAlphaBetaAgent, DeeperSearchIsStronger){
  std::mt19937_64 rng(3);
  r::RandomAgent<Game> random(1);
  r::OthelloAlphaBetaAgent shallow(1, 2), deep(4, 3);
  EXPECT_GE(r::play_match<Game>(shallow, random, 20, 2, rng).score(), 0.8);
  EXPECT_GE(r::play_match<Game>(deep, shallow, 20, 2, rng).score(), 0.7);
}
