#include <cassert>

#include <types.h>
#include <connectx.h>
#include <mcgs_search.h>

#include <iostream>

namespace s = std;
namespace r = rlbg;

constexpr ubyte RSZ = 6;
constexpr ubyte CSZ = 7;

using Game = r::ConnectXGameState<RSZ,CSZ,4>;

int main(){
  char input;

  r::MCPlayout<r::ConnectXBoard<RSZ,CSZ>,Game> playout(16);
  r::UCTSelector<
    r::ConnectXBoard<RSZ,CSZ>,
    Game
  > selector(0.1);
  r::NullCollector<r::ConnectXBoard<RSZ,CSZ>,Game> collector;

  bool is_player_move = false;
  Game game_state;
  while (not game_state.is_terminal()){
    s::cout << game_state.board() << s::endl;

    if (is_player_move){
      do {
        s::cin >> input;
        input -= '0';
      } while (not game_state.is_valid(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, r::Pt(0, input)))));

      r::Pt pt(0, input);
      game_state.apply(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, pt)));
    } else {
      r::Move move = r::mcgs_search(game_state, selector, playout, collector, 64);
      game_state.apply(r::PlayerMove(game_state.next_player(), move));
    }
    is_player_move = not is_player_move;
  }

  s::cout << game_state.board();
  s::cout << "winner=" << game_state.winner() << s::endl;
}
