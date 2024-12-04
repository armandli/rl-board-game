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

constexpr ubyte RSZ = 4;
constexpr ubyte CSZ = 4;

using Game = r::ConnectXGameState<RSZ,CSZ,3>;

int main(){
//  s::cout << "press any key to continue" << s::endl;

//  s::this_thread::sleep_for(s::chrono::milliseconds(1000));

  //initscr();
  //cbreak();
  //noecho();

  //WINDOW* win = newwin(RSZ+2, CSZ+2, 0, 0);

  char input;

//  int input = getch();

  //wborder(win, 0, 0, 0, 0, 0, 0, 0, 0);

  Game game_state;
  while (not game_state.is_terminal()){
    s::cout << game_state.board() << s::endl;

    r::Move move = r::minimax_search(game_state);
    game_state.apply(r::PlayerMove(game_state.next_player(), move));

    //s::stringstream outstream;
    //game_state.board().print(outstream);

 //    int row_id = 1;
 //    wmove(win, row_id, 1);
 //    for (char c : outstream.str()){
 //      switch (c){
 //      break; case '\n':
 //        row_id++;
 //        wmove(win, row_id, 1);
 //      break; case '|':; //ignore
 //      break; default:
 //        waddch(win, c);
 //      }
 //    }
 //    wrefresh(win);

//    if (is_player_move){
//      do {
//        s::cin >> input;
//        input -= '0';
//
//        //input = getch();
//        //input -= '0';
//      } while (not game_state.is_valid(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, r::Pt(0, input)))));
//
//      r::Pt pt(0, input);
//      game_state.apply(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, pt)));
//    } else {
//      r::Move move = r::minimax_search(game_state);
//      game_state.apply(r::PlayerMove(game_state.next_player(), move));
//    }
//    is_player_move = not is_player_move;
  }

  s::cout << game_state.board();
  s::cout << "winner=" << game_state.winner() << s::endl;

//  delwin(win);
}
