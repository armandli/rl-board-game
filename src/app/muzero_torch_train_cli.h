#ifndef RLBG_APP_MUZERO_TORCH_TRAIN_CLI_H
#define RLBG_APP_MUZERO_TORCH_TRAIN_CLI_H

// command line and run loop shared by the MuZero LibTorch training apps
#include <csignal>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <torch/torch.h>

#include <encoder/encoder.h>
#include <libtorch/training_backend.h>
#include <training/basic_training_loop.h>

namespace rlbg::app {

namespace s = std;
namespace sf = std::filesystem;
namespace r = rlbg;
namespace lt = rlbg::libtorch;
namespace t = torch;

inline s::atomic<bool> interrupted{false};

// CLI11 shows a flag's default as the value it sets, so give it explicitly
inline s::string bool_str(bool v){
  return v ? "true" : "false";
}

inline void on_signal(int){
  if (interrupted.exchange(true)) s::_Exit(130);
  s::cerr << "\n[train] interrupt: finishing current work and saving a checkpoint (again to quit now)" << s::endl;
}

// model dimensions from the encoder
template <typename Game>
void set_model_shape(lt::TrainingConfig& cfg, const r::Encoder<Game>& encoder){
  cfg.model.num_tokens = encoder.num_tokens();
  cfg.model.num_features = encoder.num_features();
  cfg.model.num_actions = encoder.num_actions();
  cfg.model.num_token_classes = encoder.num_token_classes();
}

// adds every training option to app (root names the search root, set from
// cfg.search.root); returns the evaluation group for game specific baselines
inline CLI::Option_group* add_training_options(CLI::App& app, lt::TrainingConfig& cfg, s::string& root, bool& print_config){
  root = cfg.search.root == r::MCGSConfig::Root::Gumbel ? "gumbel" : "puct";
  app.set_config("--config", "", "Read options from a TOML file");
  app.add_flag("--print-config", print_config, "Print every option as TOML and exit")->configurable(false);

  auto* run = app.add_option_group("Run");
  run->add_option("--data-dir", cfg.data_dir, "Experience, checkpoints, snapshots and logs")->capture_default_str();
  run->add_flag("--resume,!--no-resume", cfg.resume, "Continue from data-dir's checkpoint and experience")->default_str(bool_str(cfg.resume));
  run->add_option("--max-steps", cfg.max_steps, "Training steps to stop at")->capture_default_str();
  run->add_option("--max-minutes", cfg.max_minutes, "Wall clock minutes this run may take, 0 for no limit")->capture_default_str();
  run->add_option("--seed", cfg.seed, "Random seed, 0 for a random one")->capture_default_str();
  run->add_option("--device", cfg.device, "auto, cuda or cpu")->check(CLI::IsMember({"auto", "cuda", "cpu"}))->capture_default_str();
  run->add_option("--log-interval", cfg.log_interval_s, "Seconds between progress lines")->capture_default_str();

  auto* net = app.add_option_group("Model");
  net->add_option("--dims", cfg.model.dims, "Token embedding size")->capture_default_str();
  net->add_option("--heads", cfg.model.num_heads, "Attention heads")->capture_default_str();
  net->add_option("--repr-layers", cfg.model.repr_layers, "Representation network blocks")->capture_default_str();
  net->add_option("--dyn-layers", cfg.model.dyn_layers, "Dynamics network blocks")->capture_default_str();
  net->add_option("--pred-layers", cfg.model.pred_layers, "Prediction network blocks")->capture_default_str();
  net->add_option("--proj-dims", cfg.model.proj_dims, "Consistency projection size")->capture_default_str();

  auto* sp = app.add_option_group("Self-play");
  sp->add_option("--actors", cfg.num_actors, "Self-play + reanalyze threads, 0 for cores - 3")->capture_default_str();
  sp->add_option("--reanalyze-actors", cfg.reanalyze_actors, "Threads refreshing old policy targets, -1 for actors / 8")->capture_default_str();
  sp->add_option("--games-per-actor", cfg.selfplay.games_per_actor, "Games each thread plays at once")->capture_default_str();
  sp->add_option("--simulations", cfg.selfplay.num_simulations, "Simulations of a full search")->capture_default_str();
  sp->add_option("--fast-simulations", cfg.selfplay.fast_simulations, "Simulations of a fast search")->capture_default_str();
  sp->add_option("--full-search-prob", cfg.selfplay.full_search_prob, "Probability a move gets a full search")->capture_default_str();
  sp->add_option("--temperature", cfg.selfplay.temperature, "Visit count sampling temperature (puct)")->capture_default_str();
  sp->add_option("--temperature-moves", cfg.selfplay.temperature_moves, "Moves sampled with temperature (puct)")->capture_default_str();
  sp->add_option("--reanalyze-positions", cfg.selfplay.reanalyze_positions, "Positions refreshed per sampled game")->capture_default_str();
  sp->add_option("--inference-dtype", cfg.inference_dtype, "float16, bfloat16 or float32")
    ->check(CLI::IsMember({"float16", "bfloat16", "float32"}))->capture_default_str();
  sp->add_option("--latent-pool", cfg.latent_pool, "Latents kept on the GPU, 0 for twice what searches hold, -1 to copy them to the host")->capture_default_str();
  sp->add_option("--inference-max-batch", cfg.server.max_batch, "Largest batched network call")->capture_default_str();
  sp->add_option("--inference-client-fraction", cfg.server.client_fraction, "Fraction of threads a batch waits for")->capture_default_str();
  sp->add_option("--inference-max-wait-us", cfg.server.max_wait_us, "Longest wait to fill a batch")->capture_default_str();

  auto* search = app.add_option_group("Search");
  search->add_option("--root", root, "gumbel (Gumbel MuZero) or puct (AlphaZero)")->check(CLI::IsMember({"gumbel", "puct"}))->capture_default_str();
  search->add_option("--max-considered-actions", cfg.search.max_considered_actions, "Gumbel top-k at the root")->capture_default_str();
  search->add_option("--maxvisit-init", cfg.search.maxvisit_init, "Gumbel c_visit")->capture_default_str();
  search->add_option("--value-scale", cfg.search.value_scale, "Gumbel c_scale")->capture_default_str();
  search->add_flag("--rescale-values,!--no-rescale-values", cfg.search.rescale_values, "Min-max normalize Q per node")->default_str(bool_str(cfg.search.rescale_values));
  search->add_option("--puct-c", cfg.search.puct_c, "PUCT exploration constant")->capture_default_str();
  search->add_option("--dirichlet-alpha", cfg.search.dirichlet_alpha, "PUCT root noise")->capture_default_str();
  search->add_option("--dirichlet-fraction", cfg.search.dirichlet_fraction, "PUCT root noise weight")->capture_default_str();
  search->add_flag("--transpositions,!--no-transpositions", cfg.search.transpositions, "Graph search: share transposed positions")->default_str(bool_str(cfg.search.transpositions));

  auto* learn = app.add_option_group("Learner");
  learn->add_option("--batch-size", cfg.batch_size, "Positions per training step")->capture_default_str();
  learn->add_option("--unroll-steps", cfg.unroll_steps, "Dynamics unroll K")->capture_default_str();
  learn->add_flag("--augment,!--no-augment", cfg.augment, "Random board symmetry per sample")->default_str(bool_str(cfg.augment));
  learn->add_option("--replay-capacity", cfg.replay_capacity, "Positions kept for training")->capture_default_str();
  learn->add_option("--min-replay", cfg.min_replay_positions, "Positions before training starts")->capture_default_str();
  learn->add_option("--sample-reuse", cfg.sample_reuse, "Training samples per self-play position")->capture_default_str();
  learn->add_option("--selfplay-lead", cfg.selfplay_lead, "Positions self-play may run ahead of the learner")->capture_default_str();
  learn->add_option("--value-horizon", cfg.value.horizon, "Moves before a value target bootstraps from a search value, 0 for the final result only")->capture_default_str();
  learn->add_option("--value-min-horizon", cfg.value.min_horizon, "Shortest bootstrap horizon, for the oldest games")->capture_default_str();
  learn->add_option("--value-horizon-decay", cfg.value.horizon_decay_steps, "Learner steps of game age per move the horizon shrinks")->capture_default_str();
  learn->add_option("--value-max-age", cfg.value.max_value_age, "Oldest search value, in learner steps, a target may bootstrap from")->capture_default_str();
  learn->add_flag("--bf16,!--no-bf16", cfg.learner.bf16, "bfloat16 autocast for training on CUDA")->default_str(bool_str(cfg.learner.bf16));
  learn->add_option("--optimizer", cfg.learner.optimizer, "muon or adamw")->check(CLI::IsMember({"muon", "adamw"}))->capture_default_str();
  learn->add_option("--lr", cfg.learner.learning_rate, "AdamW peak learning rate")->capture_default_str();
  learn->add_option("--muon-lr", cfg.learner.muon_learning_rate, "Muon peak learning rate")->capture_default_str();
  learn->add_option("--weight-decay", cfg.learner.weight_decay, "Weight decay")->capture_default_str();
  learn->add_option("--warmup-steps", cfg.learner.warmup_steps, "Linear learning rate warmup")->capture_default_str();
  learn->add_option("--lr-decay-steps", cfg.learner.total_steps, "Cosine decay horizon")->capture_default_str();
  learn->add_option("--final-lr-fraction", cfg.learner.final_lr_fraction, "Learning rate at the end of the decay")->capture_default_str();
  learn->add_option("--grad-clip", cfg.learner.grad_clip, "Global gradient norm limit")->capture_default_str();
  learn->add_option("--dynamics-gradient-scale", cfg.learner.dynamics_gradient_scale, "Gradient scale into each dynamics step")->capture_default_str();
  learn->add_option("--policy-weight", cfg.learner.policy_weight)->capture_default_str();
  learn->add_option("--value-weight", cfg.learner.value_weight)->capture_default_str();
  learn->add_option("--reward-weight", cfg.learner.reward_weight)->capture_default_str();
  learn->add_option("--consistency-weight", cfg.learner.consistency_weight, "EfficientZero SimSiam loss")->capture_default_str();
  learn->add_option("--reconstruction-weight", cfg.learner.reconstruction_weight, "Board reconstruction from latents")->capture_default_str();
  learn->add_option("--ownership-weight", cfg.learner.ownership_weight, "Final square ownership")->capture_default_str();
  learn->add_option("--publish-interval", cfg.publish_interval, "Steps between weight pushes to self-play")->capture_default_str();
  learn->add_option("--checkpoint-interval", cfg.checkpoint_interval, "Steps between checkpoints")->capture_default_str();

  auto* ev = app.add_option_group("Evaluation");
  ev->add_flag("--evaluate,!--no-evaluate", cfg.evaluate, "Rate snapshots while training")->default_str(bool_str(cfg.evaluate));
  ev->add_option("--snapshot-interval", cfg.snapshot_interval, "Steps between evaluated snapshots")->capture_default_str();
  ev->add_option("--eval-games", cfg.evaluator.games, "Games against each older snapshot")->capture_default_str();
  ev->add_option("--eval-simulations", cfg.evaluator.simulations, "Search budget in evaluation games")->capture_default_str();
  ev->add_option("--eval-opening-moves", cfg.evaluator.opening_moves, "Random opening moves")->capture_default_str();
  ev->add_option("--eval-baseline-games", cfg.evaluator.baseline_games, "Games against each baseline")->capture_default_str();
  ev->add_option("--eval-rollout-simulations", cfg.evaluator.rollout_simulations, "UCT random playout baseline, 0 disables")->capture_default_str();
  ev->add_option("--eval-gauntlet", cfg.evaluator.gauntlet, "Older snapshots to play, counted back")->delimiter(',')->capture_default_str();
  return ev;
}

// after parsing: prints the config when asked, else trains until done or
// interrupted
template <typename Game>
int run_training(CLI::App& app, lt::TrainingConfig& cfg, const s::string& root, bool print_config, const r::Encoder<Game>& encoder,
                 s::function<void(r::BasicSnapshotEvaluator<Game>&)> evaluator_setup = {}){
  cfg.search.root = root == "gumbel" ? r::MCGSConfig::Root::Gumbel : r::MCGSConfig::Root::PUCT;

  if (print_config){
    s::cout << app.config_to_str(CLI::ConfigOutputMode::AllDefaults, true);
    return 0;
  }

  t::Device device = lt::parse_device(cfg.device);
  s::cout << "[train] LibTorch " << TORCH_VERSION << " on " << device.str() << s::endl;
  if (device.is_cuda()){
    // TF32 matmuls for the float32 parts of training
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
  }

  sf::create_directories(cfg.data_dir);
  s::ofstream(sf::path(cfg.data_dir) / "config.toml") << app.config_to_str(CLI::ConfigOutputMode::AllDefaults, false);

  s::signal(SIGINT, on_signal);
  s::signal(SIGTERM, on_signal);

  auto backend = s::make_shared<lt::TrainingBackend>(cfg);
  if (backend->latent_pool() > 0)
    s::cout << "[train] latent pool of " << backend->latent_pool() << " ("
            << backend->latent_pool() * cfg.model.num_tokens * cfg.model.dims * 2 / (1 << 20) << " MB)" << s::endl;
  r::BasicTrainingLoop<Game> loop(cfg, &encoder, backend);
  if (evaluator_setup) loop.set_evaluator_setup(evaluator_setup);
  loop.run(interrupted);
  return 0;
}

}  // namespace rlbg::app

#endif//RLBG_APP_MUZERO_TORCH_TRAIN_CLI_H
