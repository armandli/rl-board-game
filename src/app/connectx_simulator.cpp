#include <cassert>

#include <types.h>
#include <connectx.h>

#include <sstream>
#include <iostream>

#include <thread>
#include <chrono>

#include <curses.h>

namespace s = std;
namespace r = rlbg;

constexpr ubyte RSZ = 6;
constexpr ubyte CSZ = 7;

using Game = r::ConnectXGameState<RSZ,CSZ,4>;

int main(){
  s::cout << "press any key to continue" << s::endl;

  s::this_thread::sleep_for(s::chrono::milliseconds(1000));

  initscr();
  cbreak();
  noecho();

  WINDOW* win = newwin(RSZ+2, CSZ+2, 0, 0);

  int input = getch();

  wborder(win, 0, 0, 0, 0, 0, 0, 0, 0);

  Game game_state;
  while (not game_state.is_terminal()){
    s::stringstream outstream;
    game_state.board().print(outstream);

    int row_id = 1;
    wmove(win, row_id, 1);
    for (char c : outstream.str()){
      switch (c){
      break; case '\n':
        row_id++;
        wmove(win, row_id, 1);
      break; case '|':; //ignore
      break; default:
        waddch(win, c);
      }
    }
    wrefresh(win);

    do {
      input = getch();
      input -= '0';
    } while (not game_state.is_valid(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, r::Pt(0, input)))));

    r::Pt pt(0, input);
    game_state.apply(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, pt)));
  }

  delwin(win); 
}
