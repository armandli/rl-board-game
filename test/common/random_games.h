#ifndef RLBG_TEST_RANDOM_GAMES_H
#define RLBG_TEST_RANDOM_GAMES_H

#include <memory>
#include <random>
#include <vector>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/trajectory.h>

namespace rlbg::test {

namespace s = std;

// a uniformly random othello game recorded as a Trajectory, with uniform
// policy targets over the legal moves; policy_valid_prob of the moves count
// as full searches
inline s::shared_ptr<Trajectory> random_trajectory(
    const OthelloEncoder& enc, s::mt19937_64& rng, double policy_valid_prob = 1.0)
{
  auto t = s::make_shared<Trajectory>();
  t->observation_size = enc.observation_size();
  t->num_tokens = enc.num_tokens();
  t->num_actions = enc.num_actions();
  OthelloBitboard g;
  auto add_state = [&]{
    size_t f = t->features.size();
    t->features.resize(f + t->observation_size);
    enc.encode(g, t->features.data() + f);
    size_t k = t->tokens.size();
    t->tokens.resize(k + t->num_tokens);
    enc.encode_tokens(g, t->tokens.data() + k);
    t->to_play.push_back((uint8_t)g.to_play());
  };
  s::uniform_real_distribution<double> u(0., 1.);
  ActionList<OthelloBitboard::NUM_ACTIONS> legal;
  while (not g.is_terminal()){
    add_state();
    g.legal_actions(legal);
    size_t p = t->policies.size();
    t->policies.resize(p + t->num_actions, 0.f);
    for (int a : legal) t->policies[p + a] = 1.f / legal.size;
    t->policy_valid.push_back(u(rng) < policy_valid_prob ? 1 : 0);
    t->root_values.push_back(0.f);
    int a = legal[s::uniform_int_distribution<int>(0, legal.size - 1)(rng)];
    t->actions.push_back((int16_t)a);
    int mover = g.to_play();
    g.apply(a);
    t->rewards.push_back((int8_t)(g.is_terminal() ? g.outcome(mover) : 0));
    ++t->length;
  }
  add_state();
  t->final_tokens.resize(t->num_tokens);
  enc.encode_tokens_absolute(g, t->final_tokens.data());
  t->outcome = (int8_t)g.outcome(0);
  return t;
}

// the game a trajectory recorded, replayed up to state t
inline OthelloBitboard replay_to(const Trajectory& tr, int t){
  OthelloBitboard g;
  for (int i = 0; i < t; ++i) g.apply(tr.actions[i]);
  return g;
}

}  // namespace rlbg::test

#endif//RLBG_TEST_RANDOM_GAMES_H
