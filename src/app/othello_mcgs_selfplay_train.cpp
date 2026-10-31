// MuZero training for othello: self-play with Monte-Carlo graph search over a
// learned transformer model, reanalyze, a parallel learner and snapshot
// evaluation. Every component is configurable from the command line or a
// TOML file (--config); --print-config shows every option with its default.
#include <csignal>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <mlx/mlx.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <model/transformer_muzero.h>
#include <training/training_loop.h>

namespace s = std;
namespace sf = std::filesystem;
namespace mx = mlx::core;
namespace r = rlbg;

using Game = r::OthelloBitboard;

namespace {

s::atomic<bool> interrupted{false};

// CLI11 shows a flag's default as the value it sets, so give it explicitly
s::string bool_str(bool v){
  return v ? "true" : "false";
}

void on_signal(int){
  if (interrupted.exchange(true)) s::_Exit(130);
  s::cerr << "\n[train] interrupt: finishing current work and saving a checkpoint (again to quit now)" << s::endl;
}

} // namespace

int main(int argc, char** argv){
  r::OthelloEncoder encoder;
  r::TrainingLoopConfig cfg;
  r::TransformerMuZeroConfig model;
  model.num_tokens = encoder.num_tokens();
  model.num_features = encoder.num_features();
  model.num_actions = encoder.num_actions();
  model.num_token_classes = encoder.num_token_classes();
  s::string device = "auto";
  s::string root = cfg.search.root == r::MCGSConfig::Root::Gumbel ? "gumbel" : "puct";
  bool print_config = false;

  CLI::App app{"MuZero training for othello with Monte-Carlo graph search"};
  app.set_config("--config", "", "Read options from a TOML file");
  app.add_flag("--print-config", print_config, "Print every option as TOML and exit")->configurable(false);

  auto* run = app.add_option_group("Run");
  run->add_option("--data-dir", cfg.data_dir, "Experience, checkpoints, snapshots and logs")->capture_default_str();
  run->add_flag("--resume,!--no-resume", cfg.resume, "Continue from data-dir's checkpoint and experience")->default_str(bool_str(cfg.resume));
  run->add_option("--max-steps", cfg.max_steps, "Training steps to stop at")->capture_default_str();
  run->add_option("--seed", cfg.seed, "Random seed, 0 for a random one")->capture_default_str();
  run->add_option("--device", device, "auto, gpu or cpu")->check(CLI::IsMember({"auto", "gpu", "cpu"}))->capture_default_str();
  run->add_option("--log-interval", cfg.log_interval_s, "Seconds between progress lines")->capture_default_str();

  auto* net = app.add_option_group("Model");
  net->add_option("--dims", model.dims, "Token embedding size")->capture_default_str();
  net->add_option("--heads", model.num_heads, "Attention heads")->capture_default_str();
  net->add_option("--repr-layers", model.repr_layers, "Representation network blocks")->capture_default_str();
  net->add_option("--dyn-layers", model.dyn_layers, "Dynamics network blocks")->capture_default_str();
  net->add_option("--pred-layers", model.pred_layers, "Prediction network blocks")->capture_default_str();
  net->add_option("--proj-dims", model.proj_dims, "Consistency projection size")->capture_default_str();

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
  learn->add_option("--compute-dtype", cfg.learner.compute_dtype, "Training precision: bfloat16, float16 or float32")
    ->check(CLI::IsMember({"bfloat16", "float16", "float32"}))->capture_default_str();
  learn->add_option("--optimizer", cfg.learner.optimizer, "muon or adamw")->check(CLI::IsMember({"muon", "adamw"}))->capture_default_str();
  learn->add_option("--lr", cfg.learner.learning_rate, "AdamW peak learning rate")->capture_default_str();
  learn->add_option("--muon-lr", cfg.learner.muon_learning_rate, "Muon peak learning rate")->capture_default_str();
  learn->add_option("--weight-decay", cfg.learner.weight_decay, "Decoupled weight decay")->capture_default_str();
  learn->add_option("--warmup-steps", cfg.learner.warmup_steps, "Linear learning rate warmup")->capture_default_str();
  learn->add_option("--lr-decay-steps", cfg.learner.total_steps, "Cosine decay horizon")->capture_default_str();
  learn->add_option("--final-lr-fraction", cfg.learner.final_lr_fraction, "Learning rate at the end of the decay")->capture_default_str();
  learn->add_option("--grad-clip", cfg.learner.grad_clip, "Global gradient norm limit")->capture_default_str();
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

  CLI11_PARSE(app, argc, argv);
  cfg.search.root = root == "gumbel" ? r::MCGSConfig::Root::Gumbel : r::MCGSConfig::Root::PUCT;

  if (print_config){
    s::cout << app.config_to_str(CLI::ConfigOutputMode::AllDefaults, true);
    return 0;
  }

  mx::Device gpu(mx::Device::gpu);
  bool use_gpu = device == "gpu" or (device == "auto" and mx::is_available(gpu));
  mx::set_default_device(use_gpu ? gpu : mx::Device(mx::Device::cpu));
  s::cout << "[train] MLX " << mx::version() << " on " << (use_gpu ? "gpu" : "cpu") << s::endl;

  sf::create_directories(cfg.data_dir);
  s::ofstream(sf::path(cfg.data_dir) / "config.toml") << app.config_to_str(CLI::ConfigOutputMode::AllDefaults, false);

  s::signal(SIGINT, on_signal);
  s::signal(SIGTERM, on_signal);

  auto make_model = [model]{ return s::shared_ptr<r::MuZeroModel>(s::make_shared<r::TransformerMuZero>(model)); };
  r::TrainingLoop<Game> loop(cfg, &encoder, make_model);
  loop.run(interrupted);
  return 0;
}
