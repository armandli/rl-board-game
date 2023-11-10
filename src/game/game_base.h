#ifndef RLBG_GAME_BASE
#define RLBG_GAME_BASE

#include <type_alias.h>
#include <types.h>

namespace rlbg {

namespace s = std;

template <typename Sub>
struct Board {
  bool is_valid(Pt pt) const {
    return static_cast<Sub*>(this)->is_valid(pt);
  }
  Player operator[](Pt pt) const {
    return static_cast<Sub*>(this)->operator[](pt);
  }
  void place(Player player, Pt pt){
    static_cast<Sub*>(this)->place(player, pt);
  }
  size_t size() const {
    return static_cast<Sub*>(this)->size();
  }
  s::ostream& print(s::ostream& out) const {
    return static_cast<Sub*>(this)->print(out);
  }
};

template <typename Board, typename Sub>
struct GameState {
  const Board& board() const {
    return static_cast<Sub*>(this)->board();
  }
  Player next_player() const {
    return static_cast<Sub*>(this)->next_player();
  }
  bool is_valid(Move m) const {
    return static_cast<Sub*>(this)->is_valid();
  }
  s::vector<Move> legal_moves() const {
    return static_cast<Sub*>(this)->legal_moves();
  }
  GameState& apply(PlayerMove pm){
    return static_cast<Sub*>(this)->apply(pm);
  }
  bool is_terminal() const {
    return static_cast<Sub*>(this)->is_terminal();
  }
  Player winner() const {
    return static_cast<Sub*>(this)->winner();
  }
  int score(Player player) const {
    return static_cast<Sub*>(this)->score(player);
  }
  const s::vector<PlayerMove>& history() const {
    return static_cast<Sub*>(this)->history();
  }
};

} // rlbg

#endif//RLBG_GAME_BASE
