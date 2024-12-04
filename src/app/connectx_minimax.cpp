#include <cassert>

#include <types.h>
#include <connectx.h>
#include <minimax_search.h>

#include <sstream>
#include <iostream>

#include <thread>
#include <chrono>

namespace s = std;
namespace r = rlbg;

constexpr ubyte RSZ = 5;
constexpr ubyte CSZ = 5;

using Game = r::ConnectXGameState<RSZ,CSZ,3>;

int main(){
  char input;

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
      r::Move move = r::minimax_search_ab(game_state);
      game_state.apply(r::PlayerMove(game_state.next_player(), move));
    }
    is_player_move = not is_player_move;
  }

  s::cout << game_state.board();
  s::cout << "winner=" << game_state.winner() << s::endl;
}
