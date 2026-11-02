// play othello against a LibTorch MuZero model in the terminal: click a
// square to place a disc (or move the cursor with the arrow keys and press
// Enter). The model searches with Monte-Carlo graph search on a background
// thread and shows its evaluation of the position.
#include <atomic>
#include <bit>
#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <CLI/CLI.hpp>
#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/mouse.hpp>
#include <ftxui/dom/elements.hpp>
#include <torch/torch.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <libtorch/model_inference.h>
#include <search/mcgs.h>

namespace s = std;
namespace sf = std::filesystem;
namespace r = rlbg;
namespace lt = rlbg::libtorch;
namespace t = torch;
namespace fx = ftxui;

using Game = r::OthelloBitboard;

namespace {

// the model's move for a position, with its value for the side to move
struct Thinker {
  Thinker(const s::string& file, int simulations, t::Device device)
  : mSimulations(simulations)
  {
    lt::TransformerMuZeroConfig cfg = lt::load_model_config(file);
    mInference = s::make_unique<lt::ModelInference>(cfg, device, device.is_cuda() ? t::kHalf : t::kFloat,
                                                    2 * (simulations + 1) + 64);
    mInference->load_weights(file);
    mSearch = s::make_unique<r::MCGS<Game>>(&mEncoder, mInference->latent_size(), r::MCGSConfig{},
                                            (uint64_t)s::chrono::steady_clock::now().time_since_epoch().count());
  }

  r::SearchResult think(const Game& game){
    r::InferenceRequests requests;
    requests.observation_size = mInference->observation_size();
    requests.latent_size = mInference->latent_size();
    r::InferenceResults results;
    mSearch->start(game, mSimulations, false);
    while (true){
      requests.clear();
      if (not mSearch->step(requests)) break;
      mInference->infer(requests, results);
      mSearch->receive(results);
    }
    return mSearch->result(0.f);
  }

  r::OthelloEncoder                 mEncoder;
  int                               mSimulations;
  s::unique_ptr<lt::ModelInference> mInference;
  s::unique_ptr<r::MCGS<Game>>      mSearch;
};

s::string square_name(int sq){
  if (sq == Game::PASS) return "pass";
  return s::format("{}{}", (char)('a' + sq % 8), sq / 8 + 1);
}

struct State {
  Game              game;
  s::vector<Game>   history;   // positions before each human move, for undo
  int               human = 0; // 0 black (moves first), 1 white
  int               cursor = 19;
  int               last_move = -1;
  int               hover = -1;
  bool              thinking = false;
  bool              show_hints = true;
  float             model_value = 0.f; // for the model, after its last search
  bool              has_value = false;
  s::string         status;
};

} // namespace

int main(int argc, char** argv){
  s::string model = "data/othello_muzero_torch/final_model.pt";
  int simulations = 256;
  s::string color = "black";
  s::string device_name = "auto";

  CLI::App app{"Play othello against a MuZero model"};
  app.add_option("-m,--model", model, "Model file written by othello_muzero_torch_train")->capture_default_str();
  app.add_option("-s,--simulations", simulations, "Model search budget per move")->capture_default_str();
  app.add_option("-c,--color", color, "Your color: black (moves first) or white")
    ->check(CLI::IsMember({"black", "white"}))->capture_default_str();
  app.add_option("--device", device_name, "auto, cuda or cpu")->capture_default_str();
  CLI11_PARSE(app, argc, argv);

  if (not sf::exists(model)){
    s::cerr << "model " << model << " not found; train one with othello_muzero_torch_train or pass --model" << s::endl;
    return 1;
  }
  s::cout << "loading " << model << " ..." << s::endl;
  Thinker thinker(model, simulations, lt::parse_device(device_name));

  State st;
  st.human = color == "black" ? 0 : 1;
  s::mutex mutex;
  s::thread worker;
  auto screen = fx::App::Fullscreen();
  s::vector<fx::Box> boxes(64);

  auto finished_status = [&]{
    int b = s::popcount(st.game.black()), w = s::popcount(st.game.white());
    int o = st.game.outcome(st.human);
    st.status = s::format("Game over {}-{}: {}. [n] new game", b, w, o > 0 ? "you win!" : o < 0 ? "the model wins" : "draw");
  };

  // starts the model's move on a worker thread when it is its turn; the
  // model also takes forced passes, the human's are applied right away
  s::function<void()> advance = [&]{
    while (not st.game.is_terminal()){
      if (st.game.to_play() == st.human){
        if (st.game.legal() != 0){
          st.status = "Your move: click a square (or arrows + Enter)";
          return;
        }
        st.game.apply(Game::PASS);
        st.status = "You have no move and pass";
        continue;
      }
      st.thinking = true;
      st.status = s::format("Model thinking ({} simulations)...", simulations);
      if (worker.joinable()) worker.join();
      Game snapshot = st.game;
      worker = s::thread([&, snapshot]{
        r::SearchResult res = thinker.think(snapshot);
        {
          s::lock_guard<s::mutex> lock(mutex);
          // ignore a result for a position that was undone meanwhile
          if (st.thinking and st.game.hash() == snapshot.hash() and st.game.to_play() == snapshot.to_play()){
            st.game.apply(res.action);
            st.last_move = res.action;
            if (res.simulations > 0){
              st.model_value = res.root_value;
              st.has_value = true;
            }
            st.thinking = false;
          }
        }
        screen.PostEvent(fx::Event::Custom);
      });
      return;
    }
    finished_status();
  };

  auto play_human = [&](int sq){
    if (st.thinking or st.game.is_terminal() or st.game.to_play() != st.human) return;
    if (not st.game.is_legal(sq)){
      st.status = s::format("{} is not a legal move", square_name(sq));
      return;
    }
    st.history.push_back(st.game);
    st.game.apply(sq);
    st.last_move = sq;
    advance();
  };

  auto new_game = [&]{
    st.thinking = false;
    st.game = Game();
    st.history.clear();
    st.last_move = -1;
    st.has_value = false;
    advance();
  };

  auto undo = [&]{
    if (st.history.empty()) return;
    st.thinking = false;
    st.game = st.history.back();
    st.history.pop_back();
    st.last_move = -1;
    advance();
  };

  auto render_board = [&]{
    uint64_t black = st.game.black(), white = st.game.white();
    bool human_turn = not st.thinking and not st.game.is_terminal() and st.game.to_play() == st.human;
    uint64_t legal = human_turn ? st.game.legal() : 0;
    s::vector<fx::Element> rows;
    s::vector<fx::Element> header{fx::text("   ")};
    for (int c = 0; c < 8; ++c) header.push_back(fx::text(s::format("  {}  ", (char)('a' + c))) | fx::dim);
    rows.push_back(fx::hbox(s::move(header)));
    for (int row = 0; row < 8; ++row){
      s::vector<fx::Element> cells{fx::vbox({fx::text(" " + s::to_string(row + 1) + " ") | fx::dim, fx::text("   ")})};
      for (int c = 0; c < 8; ++c){
        int sq = row * 8 + c;
        bool is_black = (black >> sq) & 1, is_white = (white >> sq) & 1;
        bool is_legal = (legal >> sq) & 1;
        s::string glyph = is_black or is_white ? " ⬤ " : is_legal and st.show_hints ? " · " : "   ";
        fx::Color bg = (sq == st.hover or sq == st.cursor) and human_turn ? fx::Color::Green3
                     : sq == st.last_move ? fx::Color::DarkOliveGreen3
                     : fx::Color::Green4;
        fx::Color fg = is_black ? fx::Color::Black : is_white ? fx::Color::White : fx::Color::GrayDark;
        fx::Element cell = fx::vbox({fx::text(" " + glyph + " "), fx::text("     ")})
                         | fx::color(fg) | fx::bgcolor(bg) | fx::reflect(boxes[sq]);
        cells.push_back(cell);
      }
      rows.push_back(fx::hbox(s::move(cells)));
    }
    return fx::vbox(s::move(rows));
  };

  auto render_panel = [&]{
    int b = s::popcount(st.game.black()), w = s::popcount(st.game.white());
    s::string you = st.human == 0 ? "black" : "white";
    s::string turn = st.game.is_terminal() ? "-" : st.game.to_play() == 0 ? "black" : "white";
    s::vector<fx::Element> lines{
      fx::text("MuZero Othello") | fx::bold,
      fx::separator(),
      fx::text(s::format("You play {}", you)),
      fx::text(s::format("Black ⬤ {:>2}   White ◯ {:>2}", b, w)),
      fx::text(s::format("To move: {}", turn)),
      fx::text(s::format("Last move: {}", st.last_move < 0 ? "-" : square_name(st.last_move))),
    };
    if (st.has_value){
      float p = (st.model_value + 1.f) * 0.5f;
      lines.push_back(fx::text(s::format("Model's winning chances: {:.0f}%", 100.f * p)));
      lines.push_back(fx::gauge(p) | fx::color(fx::Color::Red));
    }
    lines.push_back(fx::separator());
    lines.push_back(fx::paragraph(st.status));
    if (st.thinking) lines.push_back(fx::spinner(2, (s::chrono::steady_clock::now().time_since_epoch().count() / 100000000) % 8));
    lines.push_back(fx::filler());
    lines.push_back(fx::separator());
    for (const char* k : {"click/Enter  place disc", "arrows       move cursor", "u            undo move",
                          "n            new game", "s            swap colors", "h            toggle hints",
                          "q / Esc      quit"})
      lines.push_back(fx::text(k) | fx::dim);
    return fx::vbox(s::move(lines)) | fx::size(fx::WIDTH, fx::EQUAL, 36);
  };

  auto ui = fx::Renderer([&]{
    s::lock_guard<s::mutex> lock(mutex);
    return fx::hbox({render_board() | fx::border, render_panel() | fx::border}) | fx::center;
  });

  ui |= fx::CatchEvent([&](fx::Event e){
    s::lock_guard<s::mutex> lock(mutex);
    if (e == fx::Event::Custom){
      if (not st.thinking) advance();
      return true;
    }
    if (e.is_mouse()){
      const fx::Mouse& m = e.mouse();
      st.hover = -1;
      for (int sq = 0; sq < 64; ++sq)
        if (boxes[sq].Contain(m.x, m.y)) st.hover = sq;
      if (m.button == fx::Mouse::Left and m.motion == fx::Mouse::Pressed and st.hover >= 0){
        st.cursor = st.hover;
        play_human(st.hover);
      }
      return true;
    }
    if (e == fx::Event::ArrowUp)    { st.cursor = (st.cursor + 56) % 64; return true; }
    if (e == fx::Event::ArrowDown)  { st.cursor = (st.cursor + 8) % 64; return true; }
    if (e == fx::Event::ArrowLeft)  { st.cursor = st.cursor / 8 * 8 + (st.cursor % 8 + 7) % 8; return true; }
    if (e == fx::Event::ArrowRight) { st.cursor = st.cursor / 8 * 8 + (st.cursor % 8 + 1) % 8; return true; }
    if (e == fx::Event::Return or e == fx::Event::Character(' ')){ play_human(st.cursor); return true; }
    // the worker owns the search until it is done; it takes the lock to
    // deliver its move, so the game must not change under it
    bool control = e == fx::Event::Character('u') or e == fx::Event::Character('n') or e == fx::Event::Character('s');
    if (control and st.thinking){
      st.status = "Wait for the model's move";
      return true;
    }
    if (e == fx::Event::Character('u')){ undo(); return true; }
    if (e == fx::Event::Character('n')){ new_game(); return true; }
    if (e == fx::Event::Character('s')){
      st.human ^= 1;
      if (not st.thinking) advance();
      return true;
    }
    if (e == fx::Event::Character('h')){ st.show_hints = not st.show_hints; return true; }
    if (e == fx::Event::Character('q') or e == fx::Event::Escape){
      st.thinking = false;
      screen.ExitLoopClosure()();
      return true;
    }
    return false;
  });

  {
    s::lock_guard<s::mutex> lock(mutex);
    advance();
  }
  // redraw while the model thinks so the spinner moves
  s::atomic<bool> running{true};
  s::thread ticker([&]{
    while (running){
      s::this_thread::sleep_for(s::chrono::milliseconds(150));
      bool think;
      {
        s::lock_guard<s::mutex> lock(mutex);
        think = st.thinking;
      }
      if (think) screen.PostEvent(fx::Event::Custom);
    }
  });
  screen.Loop(ui);
  running = false;
  ticker.join();
  if (worker.joinable()) worker.join();
  return 0;
}
