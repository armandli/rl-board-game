#ifndef RLBG_EXPERIENCE_TRAJECTORY_H
#define RLBG_EXPERIENCE_TRAJECTORY_H

#include <cstdint>

#include <mutex>
#include <vector>

namespace rlbg {

namespace s = std;

// one finished self-play game. States are indexed 0..length, actions and
// search statistics 0..length-1; state `length` is terminal.
struct Trajectory {
  int                  length = 0;
  int                  observation_size = 0;
  int                  num_tokens = 0;
  int                  num_actions = 0;

  s::vector<uint8_t>   features;     // (length + 1) * observation_size
  s::vector<uint8_t>   tokens;       // (length + 1) * num_tokens, side to move's view
  s::vector<uint8_t>   to_play;      // length + 1
  s::vector<int16_t>   actions;      // length
  s::vector<int8_t>    rewards;      // length, for the player who moved
  s::vector<float>     root_values;  // length, search value for the side to move
  s::vector<uint8_t>   final_tokens; // num_tokens, final board from player 0's view
  int8_t               outcome = 0;  // final result for player 0

  // search policies can be refreshed by reanalyze while the trajectory is
  // sampled for training, so they are guarded by policy_mutex
  s::vector<float>     policies;     // length * num_actions
  s::vector<uint8_t>   policy_valid; // length, 0 for fast (playout capped) searches
  mutable s::mutex     policy_mutex;

  uint64_t             id = 0;       // assigned by the replay buffer
  uint32_t             model_version = 0;
  uint32_t             reanalyzed = 0;

  // value target of state t: the final result for the side to move
  int value_target(int t) const {
    return to_play[t] == 0 ? outcome : -outcome;
  }
  const uint8_t* observation(int t) const { return features.data() + (size_t)t * observation_size; }
  const uint8_t* token_classes(int t) const { return tokens.data() + (size_t)t * num_tokens; }
};

}  // namespace rlbg

#endif//RLBG_EXPERIENCE_TRAJECTORY_H
