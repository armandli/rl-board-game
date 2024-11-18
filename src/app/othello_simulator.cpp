#include <cassert>

#include <types.h>
#include <othello.h>

#include <sstream>
#include <iostream>

#include <thread>
#include <chrono>

#include <curses.h>

namespace s = std;
namespace r = rlbg;

constexpr ubyte SZ = 8;

using Game = r::OthelloGameState<SZ>;

int main(){
  s::cout << "press any key to continue" << s::endl;

  s::this_thread::sleep_for(s::chrono::milliseconds(1000));

  initscr();
  cbreak();
  noecho();

  WINDOW* win = newwin(SZ+2, SZ+6, 0, 0);

  int rowi = getch(), coli;

  //wborder(win, 0, 0, 0, 0, 0, 0, 0, 0);

  Game game_state;
  while (not game_state.is_terminal()){
    s::stringstream outstream;
    game_state.board().print(outstream);

    int row_id = 1;
    wmove(win, row_id, 0);
    for (char c : outstream.str()){
      switch (c){
      break; case '\n':
        row_id++;
        wmove(win, row_id, 0);
      break; default:
        waddch(win, c);
      }
    }
    wrefresh(win);

    do {
      rowi = getch();
      coli = getch();
      rowi -= '0';
      coli -= '0';
    //TODO: can't specify pass!
    } while (not game_state.is_valid(r::PlayerMove(game_state.next_player(), r::Move(r::M::Play, r::Pt(rowi, coli))))); 

    r::Move play(r::M::Play, r::Pt(rowi, coli));
    game_state.apply(r::PlayerMove(game_state.next_player(), play));
  }

  delwin(win);
}
