#ifndef RLBG_MUZERO_GAME
#define RLBG_MUZERO_GAME

#include <array>
#include <concepts>
#include <cstdint>

namespace rlbg {

namespace s = std;

// fixed capacity list of legal action indices
template <int N>
struct ActionList {
  s::array<int16_t, N> actions;
  int                  size = 0;

  void push_back(int a){ actions[size++] = (int16_t)a; }
  int operator[](int i) const { return actions[i]; }
  const int16_t* begin() const { return actions.data(); }
  const int16_t* end() const { return actions.data() + size; }
};

// what the reinforcement learning components (encoder, search, self-play,
// evaluation) need from a game: a two player, alternating or not, zero sum
// game with a fixed discrete action space and a default constructed start
// position. Players are 0 and 1. outcome() is +1 win, 0 draw, -1 loss for the
// given player and is only meaningful once is_terminal().
template <typename G>
concept MuZeroGame = s::default_initializable<G> and s::copyable<G> and
  requires(G g, const G cg, int a, ActionList<G::NUM_ACTIONS>& list){
    { G::NUM_ACTIONS } -> s::convertible_to<int>;
    { cg.to_play() } -> s::convertible_to<int>;
    { cg.is_terminal() } -> s::convertible_to<bool>;
    { cg.outcome(a) } -> s::convertible_to<int>;
    { cg.legal_actions(list) } -> s::same_as<void>;
    { cg.hash() } -> s::convertible_to<uint64_t>;
    { g.apply(a) } -> s::same_as<void>;
  };

}  // namespace rlbg

#endif//RLBG_MUZERO_GAME
