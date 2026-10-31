#ifndef RLBG_EVALUATION_AGENTS_H
#define RLBG_EVALUATION_AGENTS_H

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <format>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <muzero_game.h>

#include <evaluation/agent.h>
#include <search/inference.h>
#include <search/search.h>

namespace rlbg {

namespace s = std;

// uniformly random legal moves; the rating anchor
template <MuZeroGame Game>
struct RandomAgent : Agent<Game> {
  explicit RandomAgent(uint64_t seed): mRng(seed) {}

  s::string name() const override { return "random"; }
  void select_actions(const s::vector<const Game*>& games, s::vector<int>& actions) override {
    actions.resize(games.size());
    ActionList<Game::NUM_ACTIONS> legal;
    for (size_t i = 0; i < games.size(); ++i){
      games[i]->legal_actions(legal);
      actions[i] = legal[s::uniform_int_distribution<int>(0, legal.size - 1)(mRng)];
    }
  }

protected:
  s::mt19937_64 mRng;
};

// plain UCT with uniformly random playouts, a fixed strength yardstick that
// knows nothing learned
template <MuZeroGame Game>
struct RolloutMCTSAgent : Agent<Game> {
  RolloutMCTSAgent(int simulations, uint64_t seed, float c = 1.f)
  : mSimulations(simulations), mC(c), mRng(seed) {}

  s::string name() const override { return s::format("mcts_rollout_{}", mSimulations); }

  void select_actions(const s::vector<const Game*>& games, s::vector<int>& actions) override {
    actions.resize(games.size());
    for (size_t i = 0; i < games.size(); ++i) actions[i] = search(*games[i]);
  }

protected:
  struct Node {
    Game  state;
    int   first_child = -1;
    int   num_children = 0;
    int   action = -1;
    int   visits = 0;
    float value = 0.f;  // total, from the perspective of the player who moved into this node
  };

  int search(const Game& root){
    mNodes.clear();
    mNodes.push_back(Node{root});
    s::vector<int> path;
    for (int it = 0; it < mSimulations; ++it){
      path.clear();
      int n = 0;
      path.push_back(n);
      while (mNodes[n].num_children > 0 and not mNodes[n].state.is_terminal()){
        n = select(n);
        path.push_back(n);
      }
      if (not mNodes[n].state.is_terminal() and mNodes[n].visits > 0){
        expand(n);
        n = mNodes[n].first_child;
        path.push_back(n);
      }
      Game g = mNodes[n].state;
      ActionList<Game::NUM_ACTIONS> legal;
      while (not g.is_terminal()){
        g.legal_actions(legal);
        g.apply(legal[s::uniform_int_distribution<int>(0, legal.size - 1)(mRng)]);
      }
      mNodes[0].visits += 1;
      for (size_t i = 1; i < path.size(); ++i){
        Node& node = mNodes[path[i]];
        node.visits += 1;
        node.value += g.outcome(mNodes[path[i - 1]].state.to_play());
      }
    }
    if (mNodes[0].num_children == 0) expand(0);
    int best = mNodes[0].first_child;
    for (int c = mNodes[0].first_child; c < mNodes[0].first_child + mNodes[0].num_children; ++c)
      if (mNodes[c].visits > mNodes[best].visits) best = c;
    return mNodes[best].action;
  }

  void expand(int n){
    ActionList<Game::NUM_ACTIONS> legal;
    mNodes[n].state.legal_actions(legal);
    int first = (int)mNodes.size();
    for (int a : legal){
      Node c{mNodes[n].state};
      c.state.apply(a);
      c.action = a;
      mNodes.push_back(c);
    }
    s::shuffle(mNodes.begin() + first, mNodes.end(), mRng);
    mNodes[n].first_child = first;
    mNodes[n].num_children = legal.size;
  }

  int select(int n){
    const Node& node = mNodes[n];
    float logn = s::log((float)node.visits + 1.f);
    int best = node.first_child;
    float best_score = -1e30f;
    for (int c = node.first_child; c < node.first_child + node.num_children; ++c){
      const Node& ch = mNodes[c];
      float score = ch.visits == 0 ? 1e30f : ch.value / ch.visits + mC * s::sqrt(logn / ch.visits);
      if (score > best_score){
        best_score = score;
        best = c;
      }
    }
    return best;
  }

  int               mSimulations;
  float             mC;
  s::mt19937_64     mRng;
  s::vector<Node>   mNodes;
};

// searches with a network, without exploration noise, batching the searches
// of all games
template <MuZeroGame Game>
struct NetworkAgent : Agent<Game> {
  NetworkAgent(
      s::string name,
      Inference* inference,
      s::function<s::unique_ptr<Search<Game>>(uint64_t)> make_search,
      int simulations)
  : mName(s::move(name)), mInference(inference), mMakeSearch(s::move(make_search)),
    mSimulations(simulations)
  {}

  s::string name() const override { return mName; }
  void set_name(const s::string& name){ mName = name; }

  void select_actions(const s::vector<const Game*>& games, s::vector<int>& actions) override {
    while (mSearches.size() < games.size()) mSearches.push_back(mMakeSearch(mSearches.size() + 1));
    for (size_t i = 0; i < games.size(); ++i) mSearches[i]->start(*games[i], mSimulations, false);
    InferenceRequests requests;
    requests.observation_size = mInference->observation_size();
    requests.latent_size = mInference->latent_size();
    InferenceResults results;
    s::vector<int> waiting;
    while (true){
      requests.clear();
      waiting.clear();
      for (size_t i = 0; i < games.size(); ++i)
        if (mSearches[i]->step(requests)) waiting.push_back((int)i);
      if (waiting.empty()) break;
      mInference->infer(requests, results);
      for (int i : waiting) mSearches[i]->receive(results);
    }
    actions.resize(games.size());
    for (size_t i = 0; i < games.size(); ++i) actions[i] = mSearches[i]->result(0.f).action;
  }

protected:
  s::string                                           mName;
  Inference*                                          mInference;
  s::function<s::unique_ptr<Search<Game>>(uint64_t)>  mMakeSearch;
  int                                                 mSimulations;
  s::vector<s::unique_ptr<Search<Game>>>              mSearches;
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_AGENTS_H
