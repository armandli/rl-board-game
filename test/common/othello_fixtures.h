#ifndef RLBG_TEST_OTHELLO_FIXTURES_H
#define RLBG_TEST_OTHELLO_FIXTURES_H

#include <cstdint>
#include <memory>
#include <vector>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/trajectory.h>
#include <model/transformer_muzero.h>
#include <search/inference.h>
#include <search/mcgs.h>
#include <search/search.h>
#include <selfplay/recorder.h>

namespace rlbg::test {

namespace s = std;

// a transformer small enough for fast tests
inline TransformerMuZeroConfig small_config(const OthelloEncoder& enc){
  TransformerMuZeroConfig c;
  c.num_tokens = enc.num_tokens();
  c.num_features = enc.num_features();
  c.num_actions = enc.num_actions();
  c.num_token_classes = enc.num_token_classes();
  c.dims = 32;
  c.num_heads = 2;
  c.repr_layers = 1;
  c.dyn_layers = 1;
  c.pred_layers = 1;
  c.proj_dims = 32;
  return c;
}

inline s::shared_ptr<TransformerMuZero> small_model(const OthelloEncoder& enc){
  return s::make_shared<TransformerMuZero>(small_config(enc));
}

// plays n self-play games at once, every search batched into one inference
// call per step, and records them
template <MuZeroGame Game>
s::vector<s::shared_ptr<Trajectory>> play_selfplay_games(
    int n, Inference& inference, const Encoder<Game>& enc, const MCGSConfig& cfg, int sims, uint64_t seed = 1)
{
  TrajectoryRecorder<Game> recorder(&enc);
  s::vector<Game> games(n);
  s::vector<s::unique_ptr<MCGS<Game>>> searches;
  s::vector<s::shared_ptr<Trajectory>> trajectories;
  for (int i = 0; i < n; ++i){
    searches.push_back(s::make_unique<MCGS<Game>>(&enc, inference.latent_size(), cfg, seed + i));
    trajectories.push_back(recorder.begin(0));
    searches[i]->start(games[i], sims, true);
  }
  InferenceRequests req;
  req.observation_size = inference.observation_size();
  req.latent_size = inference.latent_size();
  InferenceResults res;
  s::vector<int> waiting;
  while (true){
    req.clear();
    waiting.clear();
    for (int i = 0; i < n; ++i){
      while (not games[i].is_terminal()){
        if (searches[i]->step(req)){
          waiting.push_back(i);
          break;
        }
        recorder.move(*trajectories[i], games[i], searches[i]->result(0.f), true);
        if (games[i].is_terminal()) recorder.finish(*trajectories[i], games[i]);
        else                        searches[i]->start(games[i], sims, true);
      }
    }
    if (waiting.empty()) break;
    inference.infer(req, res);
    for (int i : waiting) searches[i]->receive(res);
  }
  return trajectories;
}

}  // namespace rlbg::test

#endif//RLBG_TEST_OTHELLO_FIXTURES_H
