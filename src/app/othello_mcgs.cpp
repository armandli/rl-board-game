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

  bool is_player_move = false;
  Game game_state;
  while (not game_state.is_terminal()){
    s::cout << game_state.board() << s::endl;

    if (is_player_move){
      do {
        s::cin >> rowi >> coli;
        rowi -= '0';
        coli -= '0';
      } while (not game_state.is_valid(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, r::Pt(rowi, coli)))));

      r::Pt pt(rowi, coli);
      game_state.apply(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, pt)));
    } else {
      r::Move move = r::mcgs_search(game_state, selector, playout, collector, 512);

      game_state.apply(r::PlayerMove(game_state.next_player(), move));

      if (move.mty == r::M::Pass)
        s::cout << "Pass" << s::endl;
    }
    is_player_move = not is_player_move;
  }

  s::cout << game_state.board();
  s::cout << "winner=" << game_state.winner() << s::endl;
}
