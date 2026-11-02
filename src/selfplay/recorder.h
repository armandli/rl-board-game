#ifndef RLBG_SELFPLAY_RECORDER_H
#define RLBG_SELFPLAY_RECORDER_H

#include <memory>

#include <muzero_game.h>

#include <encoder/encoder.h>
#include <experience/trajectory.h>
#include <search/search.h>

namespace rlbg {

namespace s = std;

// builds a Trajectory move by move from a game and its search results
template <MuZeroGame Game>
struct TrajectoryRecorder {
  explicit TrajectoryRecorder(const Encoder<Game>* encoder): mEncoder(encoder) {}

  s::shared_ptr<Trajectory> begin(uint32_t model_version) const {
    auto t = s::make_shared<Trajectory>();
    t->observation_size = mEncoder->observation_size();
    t->num_tokens = mEncoder->num_tokens();
    t->num_actions = mEncoder->num_actions();
    t->model_version = model_version;
    return t;
  }

  // records the state before the move, then applies it to game
  void move(Trajectory& t, Game& game, const SearchResult& result, bool policy_valid) const {
    add_state(t, game);
    t.actions.push_back((int16_t)result.action);
    t.policies.insert(t.policies.end(), result.policy.begin(), result.policy.end());
    t.policy_valid.push_back(policy_valid ? 1 : 0);
    t.root_values.push_back(result.root_value);
    t.value_versions.push_back(result.simulations > 0 ? t.model_version : Trajectory::NO_VALUE);
    int mover = game.to_play();
    game.apply(result.action);
    t.rewards.push_back((int8_t)(game.is_terminal() ? game.outcome(mover) : 0));
    ++t.length;
  }

  // records the terminal state
  void finish(Trajectory& t, const Game& game) const {
    add_state(t, game);
    t.final_tokens.resize(t.num_tokens);
    mEncoder->encode_tokens_absolute(game, t.final_tokens.data());
    t.outcome = (int8_t)game.outcome(0);
  }

protected:
  void add_state(Trajectory& t, const Game& game) const {
    size_t f = t.features.size();
    t.features.resize(f + t.observation_size);
    mEncoder->encode(game, t.features.data() + f);
    size_t k = t.tokens.size();
    t.tokens.resize(k + t.num_tokens);
    mEncoder->encode_tokens(game, t.tokens.data() + k);
    t.to_play.push_back((uint8_t)game.to_play());
  }

  const Encoder<Game>* mEncoder;
};

}  // namespace rlbg

#endif//RLBG_SELFPLAY_RECORDER_H
