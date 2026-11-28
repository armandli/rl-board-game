// MuZero training for Wall Go v1 on LibTorch: the same self-play, reanalyze,
// learner and snapshot evaluation as the othello trainer. --rules picks the
// rule set (see WallGoV1Rules in wall_go_v1.h); every other option is shared with
// the othello trainer, see --print-config.
#include <string>

#include <CLI/CLI.hpp>

#include <wall_go_v1.h>

#include <encoder/wall_go_v1_encoder.h>

#include "muzero_torch_train_cli.h"

namespace s = std;
namespace r = rlbg;
namespace ra = rlbg::app;
namespace lt = rlbg::libtorch;

namespace {

// starting point, not tuned: the othello model and schedule, with less replay
// since a Wall Go position (133 tokens x 21 features) is 11 times larger and a
// game around 2.5 times longer
lt::TrainingConfig default_config(const r::WallGoV1Encoder<>& encoder){
  lt::TrainingConfig cfg;
  cfg.data_dir = "data/wall_go_v1_muzero_torch";
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
  cfg.replay_capacity = 150000;
  cfg.min_replay_positions = 20000;
  cfg.sample_reuse = 4.;
  cfg.selfplay_lead = 20000;
  cfg.publish_interval = 50;
  cfg.snapshot_interval = 2000;
  cfg.checkpoint_interval = 1000;

  cfg.selfplay.games_per_actor = 32;
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
  r::WallGoV1Encoder<r::WALL_GO_V1_RULES> encoder;
  r::WallGoV1Encoder<r::WALL_GO_V1_TUTORIAL_RULES> tutorial_encoder;
  lt::TrainingConfig cfg = default_config(encoder);
  s::string root;
  s::string rules = "standard";
  bool print_config = false;

  CLI::App app{"MuZero training for Wall Go v1 on LibTorch with Monte-Carlo graph search"};
  ra::add_training_options(app, cfg, root, print_config);
  app.add_option("--rules", rules,
                 "standard: 4 pieces each placed in turn, move 0-2 steps, 20 walls each; "
                 "tutorial: 2 pieces each from the corners, move 1 step, unlimited walls")
    ->check(CLI::IsMember({"standard", "tutorial"}))->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  if (rules == "tutorial")
    return ra::run_training<r::WallGoV1Tutorial>(app, cfg, root, print_config, tutorial_encoder);
  return ra::run_training<r::WallGoV1>(app, cfg, root, print_config, encoder);
}
