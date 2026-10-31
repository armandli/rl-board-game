#ifndef RLBG_EVALUATION_AGENT_H
#define RLBG_EVALUATION_AGENT_H

#include <string>
#include <vector>

namespace rlbg {

namespace s = std;

// a player for evaluation matches; picks moves for many games at once so
// network agents can batch their searches
template <typename Game>
struct Agent {
  virtual ~Agent() = default;

  virtual s::string name() const = 0;
  // actions[i] is the move for games[i]; every game has the agent to move
  virtual void select_actions(const s::vector<const Game*>& games, s::vector<int>& actions) = 0;
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_AGENT_H
