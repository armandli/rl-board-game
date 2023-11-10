#include <cassert>
#include <iostream>

#include <types.h>
#include <connectx.h>

namespace s = std;
namespace r = rlbg;

using Game = r::ConnectXGameState<6,7,4>;

r::Pt read_input(const Game& state){
  int col; s::cin >> col;
  while (not state.is_valid(r::Move(r::M::Play, r::Pt(0, col)))){
    s::cin >> col;
  }
  return r::Pt(0, col);
}

int main(int argc, const char* argv[]){
  Game game_state;
  while (not game_state.is_terminal()){
    s::cout << game_state.board();
    s::cout << game_state.next_player() << " Play: ";
    r::Pt pt = read_input(game_state);
    game_state.apply(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, pt)));
  }
  s::cout << game_state.board();
  s::cout << "Winner: " << game_state.winner() << s::endl;
  return 0;
}
