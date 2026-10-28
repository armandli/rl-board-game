#include <sstream>
#include <string>
#include <vector>

#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <types.h>
#include <othello.h>

namespace s = std;
namespace r = rlbg;
namespace fx = ftxui;

constexpr ubyte SZ = 8;

using Game = r::OthelloGameState<SZ>;

fx::Element render_board(const Game& game){
  const auto& board = game.board();
  s::vector<fx::Element> rows;

  s::string hdr = "   ";
  for (uint ci = 0; ci < SZ; ++ci)
    hdr += s::to_string(ci);
  rows.push_back(fx::text(hdr));

  for (uint ri = 0; ri < SZ; ++ri){
    s::vector<fx::Element> cells;
    s::string row_label = " " + s::to_string(ri) + " ";
    cells.push_back(fx::text(row_label));
    for (uint ci = 0; ci < SZ; ++ci){
      r::Pt pt(ri, ci);
      fx::Element cell;
      switch (board[pt]){
      case r::Player::Black:
        cell = fx::text("X") | fx::color(fx::Color::Red);
        break;
      case r::Player::White:
        cell = fx::text("O") | fx::color(fx::Color::Blue);
        break;
      default:
        if (not game.is_terminal() and
            board.get_legal_moves(game.next_player()).contains(pt))
          cell = fx::text("+") | fx::color(fx::Color::Cyan);
        else
          cell = fx::text(".");
      }
      cells.push_back(cell);
    }
    cells.push_back(fx::text(" " + s::to_string(ri)));
    rows.push_back(fx::hbox(s::move(cells)));
  }

  rows.push_back(fx::text(hdr));
  return fx::vbox(s::move(rows));
}

int main(){
  Game game_state;
  s::string input_str;
  s::string status_msg;
  bool game_over = false;

  auto update_status = [&](){
    s::stringstream ss;
    ss << game_state.next_player();
    status_msg = "Player " + ss.str() + "'s turn. Enter row+col (e.g. 34) or 'p' to pass:";
  };
  update_status();

  auto app = fx::App::TerminalOutput();
  auto exit_fn = app.ExitLoopClosure();

  fx::InputOption input_opt;
  input_opt.on_enter = [&](){
    if (game_over){
      exit_fn();
      return;
    }
    if (input_str.empty()) return;

    r::PlayerMove pm(r::Player::Unknown, r::Move(r::M::Unknown));
    if (input_str == "p" or input_str == "pass"){
      pm = r::PlayerMove(game_state.next_player(), r::Move(r::M::Pass));
    } else if (input_str.size() >= 2 and
               input_str[0] >= '0' and input_str[0] <= '9' and
               input_str[1] >= '0' and input_str[1] <= '9'){
      int row = input_str[0] - '0';
      int col = input_str[1] - '0';
      pm = r::PlayerMove(
        game_state.next_player(),
        r::Move(r::M::Play, r::Pt(row, col))
      );
    } else {
      input_str.clear();
      status_msg = "Invalid input. Enter row+col (e.g. 34) or 'p' to pass:";
      return;
    }
    input_str.clear();

    if (not game_state.is_valid(pm)){
      status_msg = "Invalid move. Try again:";
      return;
    }
    game_state.apply(pm);
    if (game_state.is_terminal()){
      game_over = true;
      s::stringstream ss;
      r::Player w = game_state.winner();
      if (w == r::Player::Unknown)
        status_msg = "Draw! Press Enter to exit.";
      else {
        ss << w;
        status_msg = ss.str() + " wins! Press Enter to exit.";
      }
    } else {
      update_status();
    }
  };

  auto input_comp = fx::Input(&input_str, "rc or p", input_opt);

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
