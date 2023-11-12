#include <cassert>
#include <iostream>

#include <types.h>
#include <othello.h>

namespace s = std;
namespace r = rlbg;

using Game = r::OthelloGameState<8>;

r::Move read_input(const Game& state){
  int row, col; s::cin >> row >> col;
  if (row < 0 or col < 0) return r::Move(r::M::Pass);
  while (not state.is_valid(r::PlayerMove(state.next_player(), r::Move(r::M::Play, r::Pt(row, col))))){
    s::cout << "Invalid coordinate! Again: ";
    s::cin >> row >> col;
  }
  return r::Move(r::M::Play, r::Pt(row, col));
}

int main(int argc, const char* argv[]){
  Game game_state;
  while (not game_state.is_terminal()){
    s::cout << game_state.board();
    s::cout << game_state.next_player() << " Play: ";
    r::Move play = read_input(game_state);
    game_state.apply(r::PlayerMove(game_state.next_player(), play));
  }
  s::cout << game_state.board();
  s::cout << "Winner: " << game_state.winner() << s::endl;
  return 0;
}
