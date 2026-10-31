#ifndef RLBG_EVALUATION_ARENA_H
#define RLBG_EVALUATION_ARENA_H

#include <random>
#include <vector>

#include <muzero_game.h>

#include <evaluation/agent.h>

namespace rlbg {

namespace s = std;

// results from the first agent's point of view
struct MatchResult {
  int wins = 0;
  int draws = 0;
  int losses = 0;

  int games() const { return wins + draws + losses; }
  double score() const { return games() ? (wins + 0.5 * draws) / games() : 0.5; }
};

// plays num_games between a and b, all at once so network agents batch their
// searches. Games come in pairs from the same random opening of
// opening_moves moves with colors swapped, which removes the first move
// advantage and gives deterministic agents varied positions.
template <MuZeroGame Game>
MatchResult play_match(Agent<Game>& a, Agent<Game>& b, int num_games, int opening_moves, s::mt19937_64& rng){
  struct Seat {
    Game game;
    int  a_player;
  };
  s::vector<Seat> seats;
  ActionList<Game::NUM_ACTIONS> legal;
  while ((int)seats.size() < num_games){
    Game g;
    for (int i = 0; i < opening_moves and not g.is_terminal(); ++i){
      g.legal_actions(legal);
      g.apply(legal[s::uniform_int_distribution<int>(0, legal.size - 1)(rng)]);
    }
    if (g.is_terminal()) continue;
    seats.push_back(Seat{g, 0});
    seats.push_back(Seat{g, 1});
  }
  seats.resize(num_games);

  s::vector<const Game*> to_move;
  s::vector<int> idx;
  s::vector<int> actions;
  for (bool active = true; active;){
    active = false;
    for (Agent<Game>* agent : {&a, &b}){
      to_move.clear();
      idx.clear();
      for (int i = 0; i < (int)seats.size(); ++i){
        const Seat& st = seats[i];
        if (st.game.is_terminal()) continue;
        bool a_turn = st.game.to_play() == st.a_player;
        if (a_turn == (agent == &a)){
          to_move.push_back(&st.game);
          idx.push_back(i);
        }
      }
      if (to_move.empty()) continue;
      active = true;
      agent->select_actions(to_move, actions);
      for (size_t j = 0; j < idx.size(); ++j) seats[idx[j]].game.apply(actions[j]);
    }
  }

  MatchResult r;
  for (const Seat& st : seats){
    int o = st.game.outcome(st.a_player);
    if (o > 0)      ++r.wins;
    else if (o < 0) ++r.losses;
    else            ++r.draws;
  }
  return r;
}

}  // namespace rlbg

#endif//RLBG_EVALUATION_ARENA_H
