// MuZero training for othello on LibTorch: continuous self-play with
// Monte-Carlo graph search over a learned transformer model, reanalyze, a
// parallel learner, snapshots and their evaluation against older snapshots
// and fixed baselines. Every component is configurable from the command line
// or a TOML file (--config); --print-config shows every option with its
// default.
#include <functional>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <evaluation/othello_alpha_beta_agent.h>

#include "muzero_torch_train_cli.h"

namespace s = std;
namespace r = rlbg;
namespace ra = rlbg::app;
namespace lt = rlbg::libtorch;

using Game = r::OthelloBitboard;

namespace {

// defaults for othello on one consumer GPU (RTX 5080, 16 GB) and a 16-24
// core CPU; see README for the hyperparameter search behind them
lt::TrainingConfig default_config(const r::OthelloEncoder& encoder){
  lt::TrainingConfig cfg;
  cfg.data_dir = "data/othello_muzero_torch";
  ra::set_model_shape(cfg, encoder);
  cfg.model.dims = 192;
  cfg.model.num_heads = 6;
  cfg.model.repr_layers = 4;
  cfg.model.dyn_layers = 3;
  cfg.model.pred_layers = 1;
  cfg.model.proj_dims = 192;

  cfg.max_steps = 100000;
  cfg.batch_size = 256;
  cfg.unroll_steps = 5;
  // a short window: early games of a weak network go stale fast
  cfg.replay_capacity = 200000;
  cfg.min_replay_positions = 20000;
  cfg.sample_reuse = 4.;
  cfg.selfplay_lead = 20000;
  cfg.publish_interval = 50;
  cfg.snapshot_interval = 2000;
  cfg.checkpoint_interval = 1000;

  cfg.selfplay.games_per_actor = 32;
  // value targets from the game result; bootstrapping from search values
  // (EfficientZero) learned slower in the search
  cfg.value.horizon = 0;
  cfg.selfplay.num_simulations = 64;
  cfg.selfplay.fast_simulations = 16;
  cfg.selfplay.full_search_prob = 0.4f;
  cfg.server.max_batch = 4096;

  cfg.learner.total_steps = 100000;
  cfg.learner.warmup_steps = 1000;

  cfg.evaluator.games = 64;
  cfg.evaluator.simulations = 64;
  cfg.evaluator.baseline_games = 32;
  cfg.evaluator.rollout_simulations = 1000;
  return cfg;
}

} // namespace

int main(int argc, char** argv){
  r::OthelloEncoder encoder;
  lt::TrainingConfig cfg = default_config(encoder);
  s::string root;
  int alphabeta_depth = 4;
  bool print_config = false;

  CLI::App app{"MuZero training for othello on LibTorch with Monte-Carlo graph search"};
  auto* ev = ra::add_training_options(app, cfg, root, print_config);
  ev->add_option("--eval-alphabeta-depth", alphabeta_depth, "Alpha-beta baseline search depth, 0 disables")->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  s::function<void(r::BasicSnapshotEvaluator<Game>&)> setup;
  if (alphabeta_depth > 0)
    setup = [alphabeta_depth](r::BasicSnapshotEvaluator<Game>& evaluator){
      evaluator.add_baseline([alphabeta_depth](uint64_t seed){
        return s::unique_ptr<r::Agent<Game>>(s::make_unique<r::OthelloAlphaBetaAgent>(alphabeta_depth, seed));
      });
    };
  return ra::run_training<Game>(app, cfg, root, print_config, encoder, setup);
}
