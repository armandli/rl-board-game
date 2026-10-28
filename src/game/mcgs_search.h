#ifndef RLBG_MCGS_SEARCH
#define RLBG_MCGS_SEARCH

#include <ctime>

#include <deque>
#include <iostream>
#include <random>

#include <type_alias.h>
#include <types.h>
#include <game_base.h>

namespace s = std;

namespace rlbg {

template <typename Board, typename Sub>
struct Node {
  Node() = default;
  explicit Node(GameState<Board,Sub>* st): state(st), u(0.f), q(0.f), n(0) {}

  GameState<Board,Sub>*           state;
  umap<Move, s::pair<Node*,uint>> children;
  float                           u;
  s::vector<float>                pa;
  float                           q;
  uint                            n;
};

template <typename Board, typename Sub>
struct NullCollector {
  void collect(const Node<Board,Sub>&){}
};

template <typename Board, typename Sub>
void cleanup_state_map(umap<size_t, GameState<Board,Sub>*>& states){
  for (auto [h, st] : states)
    delete static_cast<Sub*>(st);
}

s::default_random_engine& get_default_random_engine(){
  static s::default_random_engine engine(time(0));
  return engine;
}

template <typename Board, typename Sub>
struct MCPlayout {
  uint n;
  explicit MCPlayout(uint n): n(n) {}
  s::pair<float, s::vector<float>> operator()(
      GameState<Board,Sub>* state,
      Player player)
  {
    float res = 0.;
    for (uint i = 0; i < n; ++i){
      Sub s(*static_cast<Sub*>(state));
      while (not s.is_terminal()){
        s::vector<Move> moves = s.legal_moves();

        assert(moves.size() > 0);

        s::uniform_int_distribution<uint> dist(0, moves.size()-1);
        uint idx = dist(get_default_random_engine());

        assert(idx < moves.size());

        s.apply(PlayerMove(s.next_player(), moves[idx]));
      }
      res += s.score(player);
    }

    // fake p(a)
    s::vector<Move> moves = state->legal_moves();
    s::vector<float> ps(moves.size(), 1.f);

    return s::make_pair((float)(res/n), ps);
  }
};

template <typename Board, typename Sub>
struct UCTSelector {
  float cpuct;
  explicit UCTSelector(float cpuct): cpuct(cpuct) {}
  Move operator()(Node<Board,Sub>* node, Player player){
    assert(node != nullptr);
    assert(node->state != nullptr);

    s::vector<Move> moves = node->state->legal_moves();

    assert(node->pa.size() == moves.size());
    assert(moves.size() > 0);

    s::vector<uint> mc;
    s::vector<float> mq;
    for (const Move& move : moves){
      if (node->children.find(move) == node->children.end()){
        mc.push_back(0);
        mq.push_back(0.);
      } else {
        auto [child, visit] = node->children.at(move);
        mc.push_back(visit);
        mq.push_back(child->q);
      }
    }

    uint sum_visit = 0;
    for (uint v : mc){
      sum_visit += v;
    }

    // if current state player not root player,
    // mq is minimized and need to revert the sign
    float pfactor = 1.f;
    if (player != node->state->next_player())
      pfactor = -1.f;

    // selection is always based on maximization because exploration bonus is always in positive
    float bq = -1;
    size_t bidx = 0;
    for (size_t i = 0; i < moves.size(); ++i){
      float q = pfactor * mq[i] + cpuct * node->pa[i] * sqrt((float)sum_visit) / (1.f + mc[i]);
      if (q > bq){
        bq = q;
        bidx = i;
      }
    }
    return moves[bidx];
  }
};

template <typename Board, typename Sub, typename SEL, typename PRED>
void mcgs_rollout(
    Node<Board,Sub>* node,
    Player player,
    SEL& sel,
    PRED& pred,
    umap<size_t, GameState<Board,Sub>*>& states,
    umap<size_t, Node<Board,Sub>*>& nodes,
    s::deque<Node<Board,Sub>>& node_storage
){
  assert(node != nullptr);
  assert(node->state != nullptr);

  if (node->state->is_terminal()){
    node->u = node->state->score(player);
  // 3. playout/predict
  } else if (node->u == 0.f){
    auto [u, pa] = pred(node->state, player);
    node->u = u;
    node->pa = pa;
  } else {
    // note: mcgs would crash if DAG has cycles
    Move move = sel(node, player);
    // 2. create new node
    if (node->children.find(move) == node->children.end()){
      GameState<Board,Sub>* new_state = new Sub(static_cast<Sub&>(*node->state));
      new_state->apply(PlayerMove(new_state->next_player(), move));
      size_t h = new_state->hash();
      if (nodes.find(h) != nodes.end()){
        delete static_cast<Sub*>(new_state);
        node->children[move] = {nodes[h], 0U};
      } else {
        states[h] = new_state;
        node_storage.emplace_back(new_state);
        nodes[h] = &node_storage.back();
        node->children[move] = {nodes[h], 0U};
      }
    }
    // 1. rollout
    auto& [child, vcount] = node->children[move];
    mcgs_rollout(child, player, sel, pred, states, nodes, node_storage);
    vcount++;
  }

  // 4. update
  node->n = 1;
  for (auto& [m, v] : node->children){
    auto& [child, vcount] = v;
    node->n += vcount;
  }
  node->q = node->u;
  for (auto& [m, v] : node->children){
    auto& [child, vcount] = v;
    node->q += child->q * vcount;
  }
  node->q = node->q * (1.f / node->n);
}

template <typename Board, typename Sub, typename SEL, typename PRED, typename COL>
Move mcgs_search(
    GameState<Board,Sub>& root_state,
    SEL& sel,
    PRED& pred,
    COL& collector,
    uint iterations)
{
  Player player = root_state.next_player();
  umap<size_t, GameState<Board,Sub>*> states;
  s::deque<Node<Board,Sub>> node_storage;
  umap<size_t, Node<Board,Sub>*> nodes;

  states[root_state.hash()] = new Sub(static_cast<Sub&>(root_state));
  node_storage.emplace_back(states[root_state.hash()]);
  nodes[root_state.hash()] = &node_storage.back();

  for (uint i = 0; i < iterations; ++i){
    Node<Board,Sub>* node = nodes[root_state.hash()];
    mcgs_rollout(node, player, sel, pred, states, nodes, node_storage);
  }

  //TODO: maybe we should do a whole tree stats update before selecting the move

  cleanup_state_map(states);

  Move move(M::Resign);
  Node<Board,Sub>& root_node = *nodes[root_state.hash()];
  if (root_node.children.size() == 0) return move;

  collector.collect(root_node);

  uint best_v = 0;
  for (auto& [m, v] : root_node.children){
    auto& [_, vcount] = v;
    if (vcount > best_v){
      best_v = vcount;
      move = m;
    }
  }
  return move;
}

}  // namespace rlbg

#endif//RLBG_MCGS_SEARCH
