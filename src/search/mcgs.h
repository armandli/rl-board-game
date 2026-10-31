#ifndef RLBG_SEARCH_MCGS_H
#define RLBG_SEARCH_MCGS_H

#include <cassert>
#include <cmath>
#include <cstdint>

#include <algorithm>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

#include <muzero_game.h>

#include <encoder/encoder.h>
#include <search/inference.h>
#include <search/search.h>

namespace rlbg {

namespace s = std;

struct MCGSConfig {
  enum class Root { Gumbel, PUCT };

  Root  root = Root::Gumbel;
  // Gumbel MuZero (Danihelka et al. 2022)
  int   max_considered_actions = 16;
  float maxvisit_init = 50.f;  // c_visit
  float value_scale = 0.1f;    // c_scale
  bool  rescale_values = true; // min-max normalize completed Q per node
  // AlphaZero / MuZero PUCT
  float puct_c = 1.25f;
  float dirichlet_alpha = 0.3f;
  float dirichlet_fraction = 0.25f;
  // merge nodes reached by different move orders
  bool  transpositions = true;
};

// for each of num_simulations simulations, the visit count the root action
// it visits must have; the root picks the best scored action with that
// count, which implements Gumbel-top-k followed by sequential halving
// (Danihelka et al. 2022, as in mctx)
inline s::vector<int> gumbel_considered_visits(int max_considered, int num_simulations){
  s::vector<int> out;
  if (max_considered <= 1){
    for (int i = 0; i < num_simulations; ++i) out.push_back(i);
    return out;
  }
  int log2max = (int)s::ceil(s::log2((double)max_considered));
  s::vector<int> visits(max_considered, 0);
  int considered = max_considered;
  while ((int)out.size() < num_simulations){
    int extra = s::max(1, num_simulations / (log2max * considered));
    for (int k = 0; k < extra; ++k){
      out.insert(out.end(), visits.begin(), visits.begin() + considered);
      for (int i = 0; i < considered; ++i) visits[i] += 1;
    }
    considered = s::max(2, considered / 2);
  }
  out.resize(num_simulations);
  return out;
}

// Monte-Carlo graph search over a learned MuZero model.
//
// Nodes are latent states produced by the model: the root by the
// representation network, every other node by the dynamics network applied
// to its parent's latent. The real rules are still known for board games, so
// the search follows the real game state alongside each latent to mask
// illegal actions, detect the end of the game (exact terminal reward) and
// detect transpositions: when a move order reaches a position the graph
// already holds, the edge links to the existing node instead of asking the
// dynamics network for a new latent, so every position is evaluated once.
//
// Values follow "Monte-Carlo Graph Search from First Principles" (KataGo):
// a node's value is recomputed from its children on every backup,
//   Q(n) = (U(n) + sum_e N(e) (r(e) - Q(child(e)))) / (1 + sum_e N(e))
// with U the network value and N(e) edge visits, so a node shared by several
// parents gives all of them the same, complete estimate and no information
// leaks between branches. A simulation that reaches a node through an edge
// with fewer visits than the node itself stops there and backs up the
// node's current value, which costs no network call.
//
// Root selection is either Gumbel MuZero (Gumbel-top-k sampling, sequential
// halving, improved policy target softmax(logits + sigma(completed Q)),
// deterministic interior selection), which keeps improving the policy with
// as few as 8-32 simulations, or AlphaZero PUCT with Dirichlet noise and
// visit count targets.
template <MuZeroGame Game>
struct MCGS : Search<Game> {
  MCGS(const Encoder<Game>* encoder, int latent_size, const MCGSConfig& config, uint64_t seed)
  : mEncoder(encoder), mConfig(config), mLatentSize(latent_size), mRng(seed),
    mObservation(encoder->observation_size())
  {}

  void start(const Game& root, int num_simulations, bool add_noise) override {
    mNodes.clear();
    mEdges.clear();
    mLatents.clear();
    mTable.clear();
    mNumSimulations = num_simulations;
    mSimulations = 0;
    mNoise = add_noise;
    mPendingNode = -1;
    mFinished = false;
    mForcedAction = -1;
    new_node(root);

    ActionList<Game::NUM_ACTIONS> legal;
    root.legal_actions(legal);
    assert(legal.size > 0);
    if (legal.size == 1){
      // a forced move needs no search
      mForcedAction = legal[0];
      mFinished = true;
    }
  }

  bool step(InferenceRequests& requests) override {
    if (mFinished) return false;
    if (mPendingNode >= 0) return true;
    if (not mNodes[0].expanded){
      mEncoder->encode(mNodes[0].state, mObservation.data());
      mRequest = requests.add_initial(mObservation.data());
      mPendingNode = 0;
      mPendingInitial = true;
      return true;
    }
    while (mSimulations < mNumSimulations)
      if (simulate(requests)) return true;
    mFinished = true;
    return false;
  }

  void receive(const InferenceResults& results) override {
    assert(mPendingNode >= 0);
    int idx = mPendingInitial ? results.initial_index(mRequest) : results.recurrent_index(mRequest);
    int node = mPendingNode;
    mPendingNode = -1;
    expand(node, results.latent(idx), results.logits(idx), results.values[idx]);
    if (node == 0) init_root();
    else           backup();
  }

  bool finished() const override { return mFinished; }

  SearchResult result(float temperature) override {
    SearchResult r;
    r.policy.assign(mEncoder->num_actions(), 0.f);
    if (mForcedAction >= 0){
      r.action = mForcedAction;
      r.policy[mForcedAction] = 1.f;
      return r;
    }
    const Node& root = mNodes[0];
    r.root_value = root.q;
    r.simulations = mSimulations;
    if (mConfig.root == MCGSConfig::Root::Gumbel) gumbel_result(r);
    else                                          puct_result(r, temperature);
    return r;
  }

  size_t num_nodes() const { return mNodes.size(); }

protected:
  struct Edge {
    int16_t action;
    float   logit;   // log prior over legal actions
    float   prior;
    int     child = -1;
    int     visits = 0;
    float   reward = 0.f;
  };
  struct Node {
    Game  state;
    int   edge_begin = 0;
    int   edge_count = 0;
    int   latent = -1;
    float value = 0.f;   // network value U
    float q = 0.f;
    int   visits = 0;
    bool  terminal = false;
    bool  expanded = false;
  };

  int new_node(const Game& g){
    Node n;
    n.state = g;
    n.terminal = g.is_terminal();
    if (n.terminal) n.visits = 1;
    mNodes.push_back(n);
    return (int)mNodes.size() - 1;
  }

  void expand(int n, const latent_t* latent, const float* logits, float value){
    Node& node = mNodes[n];
    node.latent = (int)(mLatents.size() / mLatentSize);
    mLatents.insert(mLatents.end(), latent, latent + mLatentSize);

    ActionList<Game::NUM_ACTIONS> legal;
    node.state.legal_actions(legal);
    node.edge_begin = (int)mEdges.size();
    node.edge_count = legal.size;
    float mx = -s::numeric_limits<float>::infinity();
    for (int a : legal) mx = s::max(mx, logits[a]);
    float sum = 0.f;
    for (int a : legal) sum += s::exp(logits[a] - mx);
    float lse = mx + s::log(sum);
    for (int a : legal){
      Edge e;
      e.action = (int16_t)a;
      e.logit = logits[a] - lse;
      e.prior = s::exp(e.logit);
      mEdges.push_back(e);
    }
    node.value = value;
    node.q = value;
    node.visits = 1;
    node.expanded = true;
  }

  void init_root(){
    Node& root = mNodes[0];
    int n = root.edge_count;
    mGumbel.assign(n, 0.f);
    mRootPrior.resize(n);
    for (int i = 0; i < n; ++i) mRootPrior[i] = mEdges[root.edge_begin + i].prior;
    if (mConfig.root == MCGSConfig::Root::Gumbel){
      if (mNoise){
        s::uniform_real_distribution<float> u(1e-10f, 1.f);
        for (float& g : mGumbel) g = -s::log(-s::log(u(mRng)));
      }
      int considered = s::min(mConfig.max_considered_actions, n);
      mSchedule = gumbel_considered_visits(considered, mNumSimulations);
    } else if (mNoise and n > 1){
      s::gamma_distribution<float> gamma(mConfig.dirichlet_alpha, 1.f);
      s::vector<float> noise(n);
      float sum = 0.f;
      for (float& x : noise) sum += (x = gamma(mRng));
      for (int i = 0; i < n; ++i)
        mRootPrior[i] = (1.f - mConfig.dirichlet_fraction) * mRootPrior[i] +
                        mConfig.dirichlet_fraction * noise[i] / s::max(sum, 1e-20f);
    }
  }

  // runs one simulation; returns true when it stopped at a new leaf that
  // needs the network, in which case receive() finishes it
  bool simulate(InferenceRequests& requests){
    mPath.clear();
    int n = 0;
    while (true){
      int e = n == 0 ? select_root() : select_interior(n);
      mPath.emplace_back(n, e);
      if (mEdges[e].child < 0){
        Game g = mNodes[n].state;
        int mover = g.to_play();
        g.apply(mEdges[e].action);
        bool terminal = g.is_terminal();
        mEdges[e].reward = terminal ? (float)g.outcome(mover) : 0.f;
        uint64_t h = 0;
        if (mConfig.transpositions and not terminal){
          h = g.hash();
          auto it = mTable.find(h);
          if (it != mTable.end()){
            mEdges[e].child = it->second;
            backup();
            return false;
          }
        }
        int c = new_node(g);
        mEdges[e].child = c;
        if (terminal){
          backup();
          return false;
        }
        if (mConfig.transpositions) mTable.emplace(h, c);
        const Node& parent = mNodes[n];
        mRequest = requests.add_recurrent(mLatents.data() + (size_t)parent.latent * mLatentSize, mEdges[e].action);
        mPendingNode = c;
        mPendingInitial = false;
        return true;
      }
      const Node& child = mNodes[mEdges[e].child];
      if (child.terminal or mEdges[e].visits < child.visits){
        backup();
        return false;
      }
      n = mEdges[e].child;
    }
  }

  void backup(){
    for (auto it = mPath.rbegin(); it != mPath.rend(); ++it){
      mEdges[it->second].visits += 1;
      update_node(it->first);
    }
    ++mSimulations;
  }

  float edge_q(const Node& parent, const Edge& e) const {
    const Node& child = mNodes[e.child];
    if (child.terminal) return e.reward;
    float sign = child.state.to_play() == parent.state.to_play() ? 1.f : -1.f;
    return e.reward + sign * child.q;
  }

  void update_node(int n){
    Node& node = mNodes[n];
    int visits = 1;
    float sum = node.value;
    for (int i = 0; i < node.edge_count; ++i){
      const Edge& e = mEdges[node.edge_begin + i];
      if (e.visits == 0) continue;
      visits += e.visits;
      sum += e.visits * edge_q(node, e);
    }
    node.visits = visits;
    node.q = sum / visits;
  }

  // Q of visited edges, the mixed value estimate for unvisited ones; also
  // returns the edges' total and maximum visits
  void completed_q(int n, const float* priors, s::vector<float>& cq, int& sum_visits, int& max_visits) const {
    const Node& node = mNodes[n];
    cq.resize(node.edge_count);
    sum_visits = 0;
    max_visits = 0;
    float sum_p = 0.f;
    float sum_pq = 0.f;
    for (int i = 0; i < node.edge_count; ++i){
      const Edge& e = mEdges[node.edge_begin + i];
      if (e.visits == 0) continue;
      float q = edge_q(node, e);
      cq[i] = q;
      sum_visits += e.visits;
      max_visits = s::max(max_visits, e.visits);
      sum_p += priors[i];
      sum_pq += priors[i] * q;
    }
    float v_mix = node.value;
    if (sum_visits > 0 and sum_p > 0.f)
      v_mix = (node.value + sum_visits * sum_pq / sum_p) / (1.f + sum_visits);
    for (int i = 0; i < node.edge_count; ++i)
      if (mEdges[node.edge_begin + i].visits == 0) cq[i] = v_mix;
  }

  // sigma(completed Q) of Gumbel MuZero
  void sigma(s::vector<float>& cq, int max_visits) const {
    if (mConfig.rescale_values){
      auto [lo, hi] = s::minmax_element(cq.begin(), cq.end());
      float l = *lo;
      float range = s::max(*hi - l, 1e-8f);
      for (float& q : cq) q = (q - l) / range;
    } else {
      for (float& q : cq) q = (q + 1.f) * 0.5f;
    }
    float scale = (mConfig.maxvisit_init + max_visits) * mConfig.value_scale;
    for (float& q : cq) q *= scale;
  }

  const float* edge_priors(int n){
    const Node& node = mNodes[n];
    if (n == 0) return mRootPrior.data();
    mPriors.resize(node.edge_count);
    for (int i = 0; i < node.edge_count; ++i) mPriors[i] = mEdges[node.edge_begin + i].prior;
    return mPriors.data();
  }

  int select_root(){
    const Node& root = mNodes[0];
    if (mConfig.root == MCGSConfig::Root::PUCT) return select_puct(0);
    int sum_visits, max_visits;
    completed_q(0, mRootPrior.data(), mScratch, sum_visits, max_visits);
    sigma(mScratch, max_visits);
    int target = mSchedule[s::min(mSimulations, (int)mSchedule.size() - 1)];
    int best = -1;
    int fallback = 0;
    float best_score = -s::numeric_limits<float>::infinity();
    float fallback_score = -s::numeric_limits<float>::infinity();
    for (int i = 0; i < root.edge_count; ++i){
      const Edge& e = mEdges[root.edge_begin + i];
      float score = mGumbel[i] + e.logit + mScratch[i];
      if (e.visits == target and score > best_score){
        best_score = score;
        best = i;
      }
      if (score > fallback_score){
        fallback_score = score;
        fallback = i;
      }
    }
    return root.edge_begin + (best >= 0 ? best : fallback);
  }

  int select_interior(int n){
    if (mConfig.root == MCGSConfig::Root::PUCT) return select_puct(n);
    const Node& node = mNodes[n];
    int sum_visits, max_visits;
    completed_q(n, edge_priors(n), mScratch, sum_visits, max_visits);
    sigma(mScratch, max_visits);
    float mx = -s::numeric_limits<float>::infinity();
    for (int i = 0; i < node.edge_count; ++i){
      mScratch[i] += mEdges[node.edge_begin + i].logit;
      mx = s::max(mx, mScratch[i]);
    }
    float sum = 0.f;
    for (float& x : mScratch) sum += (x = s::exp(x - mx));
    int best = 0;
    float best_score = -s::numeric_limits<float>::infinity();
    for (int i = 0; i < node.edge_count; ++i){
      float score = mScratch[i] / sum - (float)mEdges[node.edge_begin + i].visits / (1.f + sum_visits);
      if (score > best_score){
        best_score = score;
        best = i;
      }
    }
    return node.edge_begin + best;
  }

  int select_puct(int n){
    const Node& node = mNodes[n];
    const float* priors = edge_priors(n);
    int sum_visits, max_visits;
    completed_q(n, priors, mScratch, sum_visits, max_visits);
    float explore = mConfig.puct_c * s::sqrt((float)s::max(sum_visits, 1));
    int best = 0;
    float best_score = -s::numeric_limits<float>::infinity();
    for (int i = 0; i < node.edge_count; ++i){
      float score = mScratch[i] + explore * priors[i] / (1.f + mEdges[node.edge_begin + i].visits);
      if (score > best_score){
        best_score = score;
        best = i;
      }
    }
    return node.edge_begin + best;
  }

  void gumbel_result(SearchResult& r){
    const Node& root = mNodes[0];
    int sum_visits, max_visits;
    completed_q(0, mRootPrior.data(), mScratch, sum_visits, max_visits);
    sigma(mScratch, max_visits);
    // action: the best scored among the most visited (the last survivors of
    // sequential halving)
    int best = 0;
    float best_score = -s::numeric_limits<float>::infinity();
    for (int i = 0; i < root.edge_count; ++i){
      const Edge& e = mEdges[root.edge_begin + i];
      if (e.visits != max_visits) continue;
      float score = mGumbel[i] + e.logit + mScratch[i];
      if (score > best_score){
        best_score = score;
        best = i;
      }
    }
    r.action = mEdges[root.edge_begin + best].action;
    // improved policy: softmax(logits + sigma(completed Q))
    float mx = -s::numeric_limits<float>::infinity();
    for (int i = 0; i < root.edge_count; ++i){
      mScratch[i] += mEdges[root.edge_begin + i].logit;
      mx = s::max(mx, mScratch[i]);
    }
    float sum = 0.f;
    for (float& x : mScratch) sum += (x = s::exp(x - mx));
    for (int i = 0; i < root.edge_count; ++i)
      r.policy[mEdges[root.edge_begin + i].action] = mScratch[i] / sum;
  }

  void puct_result(SearchResult& r, float temperature){
    const Node& root = mNodes[0];
    int total = 0;
    int best = 0;
    for (int i = 0; i < root.edge_count; ++i){
      const Edge& e = mEdges[root.edge_begin + i];
      total += e.visits;
      if (e.visits > mEdges[root.edge_begin + best].visits) best = i;
    }
    for (int i = 0; i < root.edge_count; ++i){
      const Edge& e = mEdges[root.edge_begin + i];
      r.policy[e.action] = total > 0 ? (float)e.visits / total : mRootPrior[i];
    }
    if (temperature > 0.f and total > 0){
      s::vector<double> w(root.edge_count);
      for (int i = 0; i < root.edge_count; ++i)
        w[i] = s::pow((double)mEdges[root.edge_begin + i].visits, 1.0 / temperature);
      s::discrete_distribution<int> d(w.begin(), w.end());
      best = d(mRng);
    }
    r.action = mEdges[root.edge_begin + best].action;
  }

  const Encoder<Game>*         mEncoder;
  MCGSConfig                   mConfig;
  int                          mLatentSize;
  s::mt19937_64                mRng;
  s::vector<uint8_t>           mObservation;

  s::vector<Node>              mNodes;
  s::vector<Edge>              mEdges;
  s::vector<latent_t>          mLatents;
  s::unordered_map<uint64_t, int> mTable;
  s::vector<s::pair<int,int>>  mPath;

  int                          mNumSimulations = 0;
  int                          mSimulations = 0;
  bool                         mNoise = false;
  bool                         mFinished = true;
  int                          mForcedAction = -1;
  int                          mPendingNode = -1;
  bool                         mPendingInitial = false;
  int                          mRequest = 0;

  s::vector<float>             mGumbel;
  s::vector<float>             mRootPrior;
  s::vector<int>               mSchedule;
  s::vector<float>             mScratch;
  s::vector<float>             mPriors;
};

}  // namespace rlbg

#endif//RLBG_SEARCH_MCGS_H
