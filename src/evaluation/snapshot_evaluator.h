#ifndef RLBG_EVALUATION_SNAPSHOT_EVALUATOR_H
#define RLBG_EVALUATION_SNAPSHOT_EVALUATOR_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include <muzero_game.h>

#include <evaluation/agents.h>
#include <evaluation/arena.h>
#include <evaluation/elo.h>
#include <evaluation/evaluator.h>
#include <model/model_inference.h>
#include <model/muzero_model.h>
#include <search/search.h>

namespace rlbg {

namespace s = std;
namespace sf = std::filesystem;
namespace mx = mlx::core;

struct EvaluatorConfig {
  int            games = 64;              // per snapshot opponent
  int            simulations = 64;        // network agents' search budget
  int            opening_moves = 4;       // random moves before a match game
  int            baseline_games = 32;     // per baseline
  int            rollout_simulations = 1000; // UCT random playout baseline, 0 disables
  // play the snapshots this many places back; 1 is the previous one
  s::vector<int> gauntlet = {1, 4};
};

// for every new snapshot plays the snapshots `gauntlet` places back, the
// random player (rating anchor at 0) and a UCT random playout baseline, then
// refits Elo over all results. Writes dir/results.csv (every match) and
// dir/ratings.csv (current ratings), and reloads results.csv on restart.
template <MuZeroGame Game>
struct SnapshotEvaluator : Evaluator {
  using ModelFactory = s::function<s::shared_ptr<MuZeroModel>()>;
  using SearchMaker = s::function<s::unique_ptr<Search<Game>>(uint64_t)>;

  SnapshotEvaluator(
      const EvaluatorConfig& config,
      const s::string& dir,
      ModelFactory make_model,
      int observation_size,
      mx::Dtype dtype,
      SearchMaker make_search,
      uint64_t seed)
  : mConfig(config), mDir(dir), mMakeSearch(s::move(make_search)), mRng(seed)
  {
    sf::create_directories(mDir);
    mCurrent = s::make_shared<ModelInference>(make_model(), observation_size, dtype);
    mOpponent = s::make_shared<ModelInference>(make_model(), observation_size, dtype);
    load_results();
  }

  // a snapshot already evaluated in an earlier run, kept as a gauntlet opponent
  void add_existing(const s::string& name, const s::string& weights_file){
    s::lock_guard<s::mutex> lock(mMutex);
    mSnapshots.emplace_back(name, weights_file);
  }

  void submit(const s::string& name, const s::string& weights_file) override {
    {
      s::lock_guard<s::mutex> lock(mMutex);
      mQueue.emplace_back(name, weights_file);
    }
    mCv.notify_one();
  }

  s::string summary() const override {
    s::lock_guard<s::mutex> lock(mMutex);
    return mSummary;
  }

  void run(const s::atomic<bool>& stop) override {
    while (not stop.load()){
      s::pair<s::string, s::string> job;
      {
        s::unique_lock<s::mutex> lock(mMutex);
        mCv.wait_for(lock, s::chrono::milliseconds(500), [&]{ return not mQueue.empty(); });
        if (mQueue.empty()) continue;
        // only the newest matters when evaluation falls behind training
        job = mQueue.back();
        mQueue.clear();
      }
      evaluate(job.first, job.second, stop);
    }
  }

protected:
  void evaluate(const s::string& name, const s::string& file, const s::atomic<bool>& stop){
    auto t0 = s::chrono::steady_clock::now();
    mCurrent->set_weights(nn::load_arrays(file));
    NetworkAgent<Game> current(name, mCurrent.get(), mMakeSearch, mConfig.simulations);
    s::vector<s::pair<s::string, s::string>> snapshots;
    {
      s::lock_guard<s::mutex> lock(mMutex);
      snapshots = mSnapshots;
    }
    s::ostringstream line;
    line << "[eval] " << name << ":";

    for (int back : mConfig.gauntlet){
      if (stop.load()) return;
      int idx = (int)snapshots.size() - back;
      if (idx < 0) continue;
      const auto& [oname, ofile] = snapshots[idx];
      mOpponent->set_weights(nn::load_arrays(ofile));
      NetworkAgent<Game> opponent(oname, mOpponent.get(), mMakeSearch, mConfig.simulations);
      MatchResult r = play_match<Game>(current, opponent, mConfig.games, mConfig.opening_moves, mRng);
      record(name, oname, r);
      line << s::format(" vs {} {}-{}-{}", oname, r.wins, r.draws, r.losses);
    }
    if (stop.load()) return;
    RandomAgent<Game> random(mRng());
    MatchResult rr = play_match<Game>(current, random, mConfig.baseline_games, mConfig.opening_moves, mRng);
    record(name, random.name(), rr);
    line << s::format(" vs random {}-{}-{}", rr.wins, rr.draws, rr.losses);
    if (mConfig.rollout_simulations > 0 and not stop.load()){
      RolloutMCTSAgent<Game> rollout(mConfig.rollout_simulations, mRng());
      MatchResult r = play_match<Game>(current, rollout, mConfig.baseline_games, mConfig.opening_moves, mRng);
      record(name, rollout.name(), r);
      line << s::format(" vs {} {}-{}-{}", rollout.name(), r.wins, r.draws, r.losses);
    }

    s::map<s::string, double> elo = mElo.ratings("random");
    write_ratings(elo);
    double secs = s::chrono::duration<double>(s::chrono::steady_clock::now() - t0).count();
    line << s::format(" | elo {:.0f}", elo[name]);
    if (mConfig.rollout_simulations > 0)
      line << s::format(" (mcts_rollout_{} {:.0f})", mConfig.rollout_simulations,
                        elo[s::format("mcts_rollout_{}", mConfig.rollout_simulations)]);
    line << s::format(" [{:.0f}s]", secs);
    {
      s::lock_guard<s::mutex> lock(mMutex);
      mSnapshots.emplace_back(name, file);
      mSummary = line.str();
    }
    s::cout << line.str() << s::endl;
  }

  void record(const s::string& a, const s::string& b, const MatchResult& r){
    mElo.add(a, b, r);
    sf::path path = mDir / "results.csv";
    bool header = not sf::exists(path);
    s::ofstream out(path, s::ios::app);
    if (header) out << "player,opponent,wins,draws,losses\n";
    out << a << ',' << b << ',' << r.wins << ',' << r.draws << ',' << r.losses << '\n';
  }

  void load_results(){
    s::ifstream in(mDir / "results.csv");
    s::string line;
    s::getline(in, line);
    while (s::getline(in, line)){
      s::istringstream ss(line);
      s::string a, b, w, d, l;
      if (not (s::getline(ss, a, ',') and s::getline(ss, b, ',') and s::getline(ss, w, ',') and
               s::getline(ss, d, ',') and s::getline(ss, l, ','))) continue;
      mElo.add(a, b, MatchResult{s::stoi(w), s::stoi(d), s::stoi(l)});
    }
  }

  void write_ratings(const s::map<s::string, double>& elo) const {
    s::vector<s::pair<s::string, double>> sorted(elo.begin(), elo.end());
    s::ofstream out(mDir / "ratings.csv");
    out << "player,elo\n";
    for (const auto& [p, e] : sorted) out << p << ',' << s::format("{:.1f}", e) << '\n';
  }

  EvaluatorConfig                          mConfig;
  sf::path                                 mDir;
  SearchMaker                              mMakeSearch;
  s::mt19937_64                            mRng;
  s::shared_ptr<ModelInference>            mCurrent;
  s::shared_ptr<ModelInference>            mOpponent;
  EloTable                                 mElo;

  mutable s::mutex                         mMutex;
  s::condition_variable                    mCv;
  s::deque<s::pair<s::string, s::string>>  mQueue;
  s::vector<s::pair<s::string, s::string>> mSnapshots;
  s::string                                mSummary;
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_SNAPSHOT_EVALUATOR_H
