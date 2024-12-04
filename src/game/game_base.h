#ifndef RLBG_GAME_BASE
#define RLBG_GAME_BASE

#include <type_alias.h>
#include <types.h>

namespace s = std;

namespace rlbg {

template <typename Sub>
struct Board {
  bool is_valid(Player player, Pt pt) const {
    return static_cast<const Sub*>(this)->is_valid(player, pt);
  }
  Player operator[](Pt pt) const {
    return static_cast<const Sub*>(this)->operator[](pt);
  }
  void place(Player player, Pt pt){
    static_cast<Sub*>(this)->place(player, pt);
  }
  void revert(Pt pt){
    static_cast<Sub*>(this)->revert(pt);
  }
  size_t size() const {
    return static_cast<const Sub*>(this)->size();
  }
  size_t hash() const {
    return static_cast<const Sub*>(this)->hash();
  }
  s::ostream& print(s::ostream& out) const {
    return static_cast<Sub*>(this)->print(out);
  }
};

template <typename Board, typename Sub>
struct GameState {
  const Board& board() const {
    return static_cast<const Sub*>(this)->board();
  }
  Player next_player() const {
    return static_cast<const Sub*>(this)->next_player();
  }
  bool is_valid(PlayerMove pm) const {
    return static_cast<const Sub*>(this)->is_valid(pm);
  }
  s::vector<Move> legal_moves() const {
    return static_cast<const Sub*>(this)->legal_moves();
  }
  bool apply(PlayerMove pm){
    return static_cast<Sub*>(this)->apply(pm);
  }
  void rollback(int steps){
    static_cast<Sub*>(this)->rollback(steps);
  }
  bool is_terminal() const {
    return static_cast<const Sub*>(this)->is_terminal();
  }
  Player winner() const {
    return static_cast<const Sub*>(this)->winner();
  }
  int score(Player player) const {
    return static_cast<const Sub*>(this)->score(player);
  }
  size_t hash() const {
    return static_cast<const Sub*>(this)->hash();
  }
  const s::vector<PlayerMove>& history() const {
    return static_cast<const Sub*>(this)->history();
  }
};

} // rlbg

#endif//RLBG_GAME_BASE
