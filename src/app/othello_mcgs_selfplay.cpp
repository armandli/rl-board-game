#include <cassert>

#include <types.h>
#include <othello.h>
#include <mcgs_search.h>

#include <iostream>

namespace s = std;
namespace r = rlbg;

constexpr ubyte SZ = 8;

using Game = r::OthelloGameState<SZ>;

int main(){
  char rowi, coli;

  r::MCPlayout<r::OthelloBoard<SZ>,Game> playout(32);
  r::UCTSelector<
    r::OthelloBoard<SZ>,
    Game
  > selector(0.1);
  r::NullCollector<r::OthelloBoard<SZ>,Game> collector;

  Game game_state;

  while (not game_state.is_terminal()){
    s::cout << game_state.board() << s::endl;

    r::Move move = r::mcgs_search(game_state, selector, playout, collector, 512);

    s::cout << game_state.next_player() << "=" << move << s::endl;

    game_state.apply(r::PlayerMove(game_state.next_player(), move));
  }

  s::cout << game_state.board();
  s::cout << "winner=" << game_state.winner() << s::endl;
}
