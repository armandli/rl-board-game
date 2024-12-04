#ifndef RLBG_MINIMAX_SEARCH
#define RLBG_MINIMAX_SEARCH

#include <type_alias.h>
#include <types.h>
#include <game_base.h>

#include <algorithm>

#include <iostream>

namespace s = std;

namespace rlbg {

template <typename Board, typename Sub>
Move minimax_search(GameState<Board,Sub>& state){
  struct Recursion {
    GameState<Board,Sub>& state;
    Player player;
    Recursion(GameState<Board,Sub>& st, Player player): state(st), player(player) {}
    s::pair<int, Move> operator()(bool is_maximize){
      if (state.is_terminal()){
        return s::make_pair(state.score(player), Move(M::Unknown));
      }

      s::vector<Move> moves = state.legal_moves();
      s::vector<int> scores;
      for (const Move& move : moves){
        state.apply(PlayerMove(state.next_player(), move));

        auto [score, _] = (*this).operator()(not is_maximize);

        scores.push_back(score);

        state.rollback(1);
      }

      if (is_maximize){
        auto iter = s::max_element(s::begin(scores), s::end(scores));
        int dist = iter - s::begin(scores);
        return s::make_pair((*iter), moves[dist]);
      } else {
        auto iter = s::min_element(s::begin(scores), s::end(scores));
        int dist = iter - s::begin(scores);
        return s::make_pair((*iter), moves[dist]);
      }
    }
  } recursion(state, state.next_player());

  auto [score, move] = recursion(true);
  return move;
}

template <typename Board, typename Sub>
Move minimax_search_ab(GameState<Board,Sub>& state){
  struct Recursion {
    GameState<Board,Sub>& state;
    Player player;
    Recursion(GameState<Board,Sub>& st, Player player): state(st), player(player) {}
    s::pair<int, Move> operator()(bool is_maximize, int alpha, int beta){
      if (state.is_terminal())
        return s::make_pair(state.score(player), Move(M::Unknown));

      s::vector<Move> moves = state.legal_moves();

      if (is_maximize){
        int bscore = -1;
        int bidx = -1;
        for (int i = 0; i < moves.size(); ++i){
          bool res = state.apply(PlayerMove(state.next_player(), moves[i]));
          if (res){
            auto [score, _] = (*this).operator()(not is_maximize, alpha, beta);
            state.rollback(1);
            if (score >= bscore){
              bscore = score;
              bidx = i;
            }
            alpha = s::max(alpha, bscore);
            if (beta <= alpha)
              break;
          }
        }
        return s::make_pair(bscore, moves[bidx]);
      } else {
        int bscore = 1;
        int bidx = -1;
        for (int i = 0; i < moves.size(); ++i){
          bool res = state.apply(PlayerMove(state.next_player(), moves[i]));
          if (res){
            auto [score, _] = (*this).operator()(not is_maximize, alpha, beta);
            state.rollback(1);
            if (score <= bscore){
              bscore = score;
              bidx = i;
            }
            beta = s::min(beta, bscore);
            if (beta <= alpha)
              break;
          }
        }
        return s::make_pair(bscore, moves[bidx]);
      }
    }
  } recursion(state, state.next_player());

  auto [score, move] = recursion(true, -1, 1);
  return move;
}

} // rlbg

#endif//RLBG_MINIMAX_SEARCH
