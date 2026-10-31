#include <cmath>
#include <format>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <evaluation/agents.h>
#include <evaluation/arena.h>
#include <search/mcgs.h>

#include <common/tictactoe.h>
#include <common/true_model.h>

namespace r = rlbg;
namespace rt = rlbg::test;

using Othello = r::OthelloBitboard;
using TTT = rt::TicTacToe;

namespace {

// runs a search to the end against a model; returns the number of requests
template <typename Game>
int run(r::Search<Game>& search, r::Inference& model){
  r::InferenceRequests req;
  req.observation_size = model.observation_size();
  req.latent_size = model.latent_size();
  r::InferenceResults res;
  int requests = 0;
  while (true){
    req.clear();
    if (not search.step(req)) break;
    EXPECT_EQ(req.size(), 1);
    model.infer(req, res);
    search.receive(res);
    ++requests;
  }
  EXPECT_TRUE(search.finished());
  return requests;
}

r::MCGSConfig config(r::MCGSConfig::Root root, bool transpositions = true){
  r::MCGSConfig c;
  c.root = root;
  c.transpositions = transpositions;
  return c;
}

TTT play(std::initializer_list<int> moves){
  TTT g;
  for (int a : moves) g.apply(a);
  return g;
}

const r::MCGSConfig::Root ROOTS[] = {r::MCGSConfig::Root::Gumbel, r::MCGSConfig::Root::PUCT};

std::string root_name(r::MCGSConfig::Root root){
  return root == r::MCGSConfig::Root::Gumbel ? "gumbel" : "puct";
}

} // namespace

TEST(GumbelConsideredVisits, SequentialHalvingSchedule){
  EXPECT_EQ(r::gumbel_considered_visits(4, 8), (std::vector<int>{0, 0, 0, 0, 1, 1, 2, 2}));
  EXPECT_EQ(r::gumbel_considered_visits(1, 4), (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(r::gumbel_considered_visits(16, 3), (std::vector<int>{0, 0, 0}));
  for (int m : {2, 5, 16})
    for (int n : {1, 7, 64, 200}){
      std::vector<int> seq = r::gumbel_considered_visits(m, n);
      ASSERT_EQ((int)seq.size(), n);
      // the first m simulations visit m distinct actions once
      for (int i = 0; i < std::min(m, n); ++i) EXPECT_EQ(seq[i], 0);
    }
}

TEST(MCGS, ForcedMoveNeedsNoSearch){
  rt::TicTacToeEncoder enc;
  rt::TicTacToeTrueModel model;
  // one empty square left
  TTT g = play({0, 1, 2, 4, 3, 5, 7, 6});
  for (auto root : ROOTS){
    r::MCGS<TTT> search(&enc, model.latent_size(), config(root), 1);
    search.start(g, 50, true);
    EXPECT_TRUE(search.finished());
    EXPECT_EQ(run(search, model), 0);
    r::SearchResult res = search.result(1.f);
    EXPECT_EQ(res.action, 8);
    EXPECT_EQ(res.policy[8], 1.f);
    EXPECT_EQ(res.simulations, 0);
  }
}

TEST(MCGS, ResultIsAValidPolicy){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  Othello g;
  g.apply(19);
  for (auto root : ROOTS)
    for (bool noise : {false, true}){
      r::MCGS<Othello> search(&enc, model.latent_size(), config(root), 7);
      search.start(g, 32, noise);
      int requests = run(search, model);
      EXPECT_LE(requests, 33) << "one root evaluation plus at most one per simulation";
      r::SearchResult res = search.result(0.f);
      EXPECT_TRUE(g.is_legal(res.action));
      EXPECT_EQ(res.simulations, 32);
      EXPECT_LE(std::abs(res.root_value), 1.f);
      double sum = 0.;
      for (int a = 0; a < 65; ++a){
        EXPECT_GE(res.policy[a], 0.f);
        if (res.policy[a] > 0.f) EXPECT_TRUE(g.is_legal(a)) << root_name(root) << " action " << a;
        sum += res.policy[a];
      }
      EXPECT_NEAR(sum, 1., 1e-4);
    }
}

TEST(MCGS, DeterministicWithoutNoise){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  Othello g;
  for (auto root : ROOTS){
    r::MCGS<Othello> a(&enc, model.latent_size(), config(root), 1);
    r::MCGS<Othello> b(&enc, model.latent_size(), config(root), 2);
    a.start(g, 24, false);
    b.start(g, 24, false);
    run(a, model);
    run(b, model);
    r::SearchResult ra = a.result(0.f), rb = b.result(0.f);
    EXPECT_EQ(ra.action, rb.action);
    EXPECT_EQ(ra.policy, rb.policy);
  }
}

TEST(MCGS, GumbelNoiseVariesTheChoice){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  Othello g;
  std::set<int> actions;
  for (int seed = 0; seed < 20; ++seed){
    r::MCGS<Othello> search(&enc, model.latent_size(), config(r::MCGSConfig::Root::Gumbel), seed);
    search.start(g, 4, true);
    run(search, model);
    actions.insert(search.result(0.f).action);
  }
  EXPECT_GT(actions.size(), 1u);
}

TEST(MCGS, PuctTemperatureSamplesVisits){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  Othello g;
  std::set<int> greedy, sampled;
  for (int seed = 0; seed < 20; ++seed){
    r::MCGS<Othello> search(&enc, model.latent_size(), config(r::MCGSConfig::Root::PUCT), seed);
    search.start(g, 16, false);
    run(search, model);
    greedy.insert(search.result(0.f).action);
    sampled.insert(search.result(1.f).action);
  }
  EXPECT_EQ(greedy.size(), 1u);
  EXPECT_GT(sampled.size(), 1u);
}

TEST(MCGS, TakesAWinAndBlocksALoss){
  rt::TicTacToeEncoder enc;
  rt::TicTacToeTrueModel model;
  // X to move with 0 1 taken: 2 wins
  TTT win = play({0, 3, 1, 4});
  // X to move must block O's 3 4 at 5
  TTT block = play({0, 3, 8, 4});
  for (auto root : ROOTS){
    for (auto [g, expected] : {std::pair{win, 2}, std::pair{block, 5}}){
      r::MCGS<TTT> search(&enc, model.latent_size(), config(root), 3);
      search.start(g, 200, false);
      run(search, model);
      EXPECT_EQ(search.result(0.f).action, expected) << root_name(root);
    }
  }
}

TEST(MCGS, WinningPositionHasAWinningValue){
  rt::TicTacToeEncoder enc;
  rt::TicTacToeTrueModel model;
  TTT win = play({0, 3, 1, 4});
  r::MCGS<TTT> search(&enc, model.latent_size(), config(r::MCGSConfig::Root::PUCT), 3);
  search.start(win, 200, false);
  run(search, model);
  EXPECT_GT(search.result(0.f).root_value, 0.5f);
}

// tic-tac-toe reaches positions by many move orders: the graph shares them
TEST(MCGS, TranspositionsShareNodesAndEvaluations){
  rt::TicTacToeEncoder enc;
  for (auto root : ROOTS){
    rt::TicTacToeTrueModel graph_model, tree_model;
    r::MCGS<TTT> graph(&enc, graph_model.latent_size(), config(root, true), 1);
    r::MCGS<TTT> tree(&enc, tree_model.latent_size(), config(root, false), 1);
    graph.start(TTT(), 800, false);
    tree.start(TTT(), 800, false);
    int g = run(graph, graph_model);
    int t = run(tree, tree_model);
    EXPECT_LT(g, t) << root_name(root);
    EXPECT_LT(graph.num_nodes(), tree.num_nodes()) << root_name(root);
  }
}

// with no value function the search sees only terminal rewards, so deep
// forks are occasionally missed: losses are rare, not impossible
TEST(MCGS, RarelyLosesTicTacToeToRandom){
  rt::TicTacToeEncoder enc;
  rt::TicTacToeTrueModel model;
  std::mt19937_64 rng(5);
  for (auto root : ROOTS){
    auto make = [&enc, &model, root](uint64_t seed){
      return std::unique_ptr<r::Search<TTT>>(std::make_unique<r::MCGS<TTT>>(&enc, model.latent_size(), config(root), seed));
    };
    r::NetworkAgent<TTT> search(root_name(root), &model, make, 300);
    r::RandomAgent<TTT> random(6);
    r::MatchResult m = r::play_match<TTT>(search, random, 40, 0, rng);
    EXPECT_LE(m.losses, 2) << root_name(root);
    EXPECT_GE(m.wins, 32) << root_name(root);
  }
}

// with a meaningful value function more simulations play better othello
TEST(MCGS, MoreSimulationsPlayStronger){
  r::OthelloEncoder enc;
  rt::OthelloTrueModel model;
  std::mt19937_64 rng(11);
  for (auto root : ROOTS){
    auto agent = [&](int sims){
      auto make = [&enc, &model, root](uint64_t seed){
        return std::unique_ptr<r::Search<Othello>>(
          std::make_unique<r::MCGS<Othello>>(&enc, model.latent_size(), config(root), seed));
      };
      return r::NetworkAgent<Othello>(std::format("{}_{}", root_name(root), sims), &model, make, sims);
    };
    auto strong = agent(96);
    auto weak = agent(8);
    r::RandomAgent<Othello> random(3);
    EXPECT_GT(r::play_match<Othello>(strong, random, 20, 4, rng).score(), 0.85) << strong.name();
    EXPECT_GT(r::play_match<Othello>(strong, weak, 20, 4, rng).score(), 0.6) << strong.name();
  }
}
