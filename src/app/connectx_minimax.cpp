#include <sstream>
#include <string>
#include <vector>

#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <types.h>
#include <connectx.h>
#include <minimax_search.h>

namespace s = std;
namespace r = rlbg;
namespace fx = ftxui;

constexpr ubyte RSZ = 5;
constexpr ubyte CSZ = 5;

using Game = r::ConnectXGameState<RSZ,CSZ,3>;

fx::Element render_board(const Game& game){
  const auto& board = game.board();
  s::vector<fx::Element> rows;
  for (int ri = (int)RSZ-1; ri >= 0; --ri){
    s::vector<fx::Element> cells;
    cells.push_back(fx::text("|"));
    for (uint ci = 0; ci < CSZ; ++ci){
      switch (board[r::Pt(ri, ci)]){
      case r::Player::Black:
        cells.push_back(fx::text("X") | fx::color(fx::Color::Red));
        break;
      case r::Player::White:
        cells.push_back(fx::text("O") | fx::color(fx::Color::Blue));
        break;
      default:
        cells.push_back(fx::text(" "));
      }
    }
    cells.push_back(fx::text("|"));
    rows.push_back(fx::hbox(s::move(cells)));
  }
  s::string col_line = " ";
  for (uint ci = 0; ci < CSZ; ++ci)
    col_line += s::to_string(ci);
  rows.push_back(fx::text(col_line));
  return fx::vbox(s::move(rows));
}

int main(){
  Game game_state;
  s::string input_str;
  s::string status_msg;
  bool game_over = false;

  // AI (Black) goes first
  {
    r::Move ai_move = r::minimax_search_ab(game_state);
    game_state.apply(r::PlayerMove(game_state.next_player(), ai_move));
  }

  if (game_state.is_terminal()){
    game_over = true;
    s::stringstream ss;
    ss << game_state.winner();
    status_msg = ss.str() + " wins! Press Enter to exit.";
  } else {
    status_msg = "Your turn (White). Enter column (0-4):";
  }

  auto app = fx::App::TerminalOutput();
  auto exit_fn = app.ExitLoopClosure();

  fx::InputOption input_opt;
  input_opt.on_enter = [&](){
    if (game_over){
      exit_fn();
      return;
    }
    if (input_str.empty()) return;

    char ch = input_str[0];
    input_str.clear();

    if (ch < '0' or ch > '9'){
      status_msg = "Invalid input. Enter a column number (0-4):";
      return;
    }
    int col = ch - '0';
    r::PlayerMove pm(
      game_state.next_player(),
      r::Move(r::M::Play, r::Pt(0, col))
    );
    if (not game_state.is_valid(pm)){
      status_msg = "Column " + s::to_string(col) + " is full or out of range. Try again:";
      return;
    }
    game_state.apply(pm);

    if (game_state.is_terminal()){
      game_over = true;
    } else {
      r::Move ai_move = r::minimax_search_ab(game_state);
      game_state.apply(r::PlayerMove(game_state.next_player(), ai_move));
      if (game_state.is_terminal())
        game_over = true;
    }

    if (game_over){
      s::stringstream ss;
      r::Player w = game_state.winner();
      if (w == r::Player::Unknown)
        status_msg = "Draw! Press Enter to exit.";
      else {
        ss << w;
        status_msg = ss.str() + " wins! Press Enter to exit.";
      }
    } else {
      status_msg = "Your turn (White). Enter column (0-4):";
    }
  };

  auto input_comp = fx::Input(&input_str, "column", input_opt);

  auto renderer = fx::Renderer(input_comp, [&]() -> fx::Element {
    return fx::vbox({
      render_board(game_state),
      fx::separator(),
      fx::text(status_msg),
      fx::hbox({fx::text("> "), input_comp->Render()}),
    });
  });

  app.Loop(renderer);
}
