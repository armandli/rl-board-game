// plays LibTorch MuZero othello models against fixed opponents or each other
// and reports the score and the Elo difference it implies. Used to compare
// training runs on the same yardstick, and to check that the snapshots of a
// run keep improving:
//
//   othello_muzero_torch_arena --model a.pt --opponent alphabeta:4 --games 200
//   othello_muzero_torch_arena --snapshots data/run/snapshots --every 4 --opponent rollout:1000
//   othello_muzero_torch_arena --model new.pt --opponent model:old.pt
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <torch/torch.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <evaluation/agents.h>
#include <evaluation/arena.h>
#include <evaluation/othello_alpha_beta_agent.h>
#include <libtorch/model_inference.h>
#include <search/mcgs.h>

namespace s = std;
namespace sf = std::filesystem;
namespace r = rlbg;
namespace lt = rlbg::libtorch;
namespace t = torch;

using Game = r::OthelloBitboard;

namespace {

struct Network {
  s::shared_ptr<lt::ModelInference>  inference;
  s::unique_ptr<r::NetworkAgent<Game>> agent;
};

Network load_network(const s::string& file, const r::OthelloEncoder& enc, const r::MCGSConfig& search, int games,
                     int simulations, t::Device device){
  lt::TransformerMuZeroConfig cfg = lt::load_model_config(file);
  int64_t pool = 2 * ((int64_t)games * (simulations + 1) + 4096);
  Network n;
  n.inference = s::make_shared<lt::ModelInference>(cfg, device, device.is_cuda() ? t::kHalf : t::kFloat, pool);
  n.inference->load_weights(file);
  int latent = n.inference->latent_size();
  auto make_search = [&enc, latent, search](uint64_t seed){
    return s::unique_ptr<r::Search<Game>>(s::make_unique<r::MCGS<Game>>(&enc, latent, search, seed));
  };
  n.agent = s::make_unique<r::NetworkAgent<Game>>(sf::path(file).stem().string(), n.inference.get(), make_search,
                                                  simulations);
  return n;
}

// "alphabeta:D", "rollout:N", "random" or "model:FILE"
s::unique_ptr<r::Agent<Game>> make_baseline(const s::string& spec, uint64_t seed){
  auto arg = [&]{ return s::stoi(spec.substr(spec.find(':') + 1)); };
  if (spec.starts_with("alphabeta:")) return s::make_unique<r::OthelloAlphaBetaAgent>(arg(), seed);
  if (spec.starts_with("rollout:"))   return s::make_unique<r::RolloutMCTSAgent<Game>>(arg(), seed);
  if (spec == "random")               return s::make_unique<r::RandomAgent<Game>>(seed);
  throw s::invalid_argument("unknown opponent " + spec);
}

double elo_diff(double score, int games){
  // half a game of slack keeps a perfect score finite
  double sc = s::clamp(score, 0.5 / games, 1. - 0.5 / games);
  return 400. * s::log10(sc / (1. - sc));
}

} // namespace

int main(int argc, char** argv){
  s::vector<s::string> models;
  s::string snapshots;
  int every = 1;
  s::vector<s::string> opponents = {"alphabeta:4"};
  int games = 100;
  int simulations = 64;
  int opening_moves = 4;
  uint64_t seed = 1;
  s::string csv;
  s::string device_name = "auto";

  CLI::App app{"Othello arena for LibTorch MuZero models"};
  app.add_option("--model", models, "Model files to rate")->delimiter(',');
  app.add_option("--snapshots", snapshots, "Rate every snapshot (step_*.pt) of this directory");
  app.add_option("--every", every, "With --snapshots, every n-th snapshot (always the last)")->capture_default_str();
  app.add_option("--opponent", opponents, "alphabeta:DEPTH, rollout:SIMULATIONS, random or model:FILE")
    ->delimiter(',')->capture_default_str();
  app.add_option("--games", games, "Games per pairing, in color swapped pairs")->capture_default_str();
  app.add_option("--simulations", simulations, "Search budget of the networks")->capture_default_str();
  app.add_option("--opening-moves", opening_moves, "Random moves before each game pair")->capture_default_str();
  app.add_option("--seed", seed)->capture_default_str();
  app.add_option("--csv", csv, "Append results to this CSV file");
  app.add_option("--device", device_name, "auto, cuda or cpu")->capture_default_str();
  CLI11_PARSE(app, argc, argv);

  if (not snapshots.empty()){
    s::vector<s::string> found;
    for (const auto& e : sf::directory_iterator(snapshots))
      if (e.path().extension() == ".pt" and e.path().filename().string().starts_with("step_"))
        found.push_back(e.path().string());
    s::sort(found.begin(), found.end());
    for (size_t i = 0; i < found.size(); ++i)
      if (i % every == 0 or i + 1 == found.size()) models.push_back(found[i]);
  }
  if (models.empty()){
    s::cerr << "nothing to rate: give --model or --snapshots" << s::endl;
    return 1;
  }

  t::Device device = lt::parse_device(device_name);
  r::OthelloEncoder enc;
  r::MCGSConfig search;
  s::mt19937_64 rng(seed);
  bool header = not csv.empty() and not sf::exists(csv);
  s::ofstream out;
  if (not csv.empty()) out.open(csv, s::ios::app);
  if (header) out << "model,opponent,games,simulations,wins,draws,losses,score,elo_diff\n";

  for (const s::string& file : models){
    Network net = load_network(file, enc, search, games, simulations, device);
    for (const s::string& spec : opponents){
      Network other;
      s::unique_ptr<r::Agent<Game>> baseline;
      r::Agent<Game>* opponent = nullptr;
      if (spec.starts_with("model:")){
        other = load_network(spec.substr(6), enc, search, games, simulations, device);
        opponent = other.agent.get();
      } else {
        baseline = make_baseline(spec, rng());
        opponent = baseline.get();
      }
      // the same openings for every model, so ratings compare
      s::mt19937_64 match_rng(seed);
      r::MatchResult res = r::play_match<Game>(*net.agent, *opponent, games, opening_moves, match_rng);
      double score = res.score();
      s::string name = sf::path(file).stem().string();
      s::cout << s::format("{:<22} vs {:<20} {:>4}-{:>3}-{:<4} score {:.3f} elo {:+.0f}", name, opponent->name(),
                           res.wins, res.draws, res.losses, score, elo_diff(score, games)) << s::endl;
      if (out.is_open())
        out << s::format("{},{},{},{},{},{},{},{:.4f},{:.1f}\n", file, opponent->name(), games, simulations, res.wins,
                         res.draws, res.losses, score, elo_diff(score, games)) << s::flush;
    }
  }
  return 0;
}
