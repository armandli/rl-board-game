#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <selfplay/inference_server.h>

#include <common/true_model.h>

namespace r = rlbg;
namespace rt = rlbg::test;

namespace {

// requests unique to a client: initial observations of positions along a
// game, recurrent requests on a latent of the client's own position
r::InferenceRequests client_requests(int client, const r::OthelloEncoder& enc, rt::OthelloTrueModel& model){
  r::InferenceRequests req;
  req.observation_size = enc.observation_size();
  req.latent_size = model.latent_size();
  r::OthelloBitboard g;
  r::ActionList<65> legal;
  std::vector<uint8_t> obs(enc.observation_size());
  for (int i = 0; i <= client % 5; ++i){
    g.legal_actions(legal);
    g.apply(legal[(client + i) % legal.size]);
  }
  for (int i = 0; i < 1 + client % 3; ++i){
    enc.encode(g, obs.data());
    req.add_initial(obs.data());
  }
  r::InferenceResults res;
  model.infer(req, res);
  g.legal_actions(legal);
  for (int i = 0; i < 2 + client % 4; ++i) req.add_recurrent(res.latent(0), legal[i % legal.size]);
  return req;
}

bool same_results(const r::InferenceResults& a, const r::InferenceResults& b){
  return a.num_initial == b.num_initial and a.latents.size() == b.latents.size() and
         std::equal(a.latents.begin(), a.latents.end(), b.latents.begin(),
                    [](r::latent_t x, r::latent_t y){ return (float)x == (float)y; }) and
         a.values == b.values and a.policy_logits == b.policy_logits and a.rewards == b.rewards;
}

} // namespace

// many clients at once get exactly what they would get alone
TEST(InferenceServer, MergesAndScattersClientBatches){
  r::OthelloEncoder enc;
  auto backend = std::make_shared<rt::OthelloTrueModel>();
  r::InferenceServerConfig cfg;
  cfg.client_fraction = 1.0;
  cfg.max_wait_us = 20000;
  r::InferenceServer server(backend, cfg);
  std::thread serving([&]{ server.run(); });

  constexpr int CLIENTS = 8;
  constexpr int ROUNDS = 20;
  for (int c = 0; c < CLIENTS; ++c) server.add_client();
  std::atomic<int> mismatches{0};
  std::vector<std::thread> clients;
  for (int c = 0; c < CLIENTS; ++c)
    clients.emplace_back([&, c]{
      rt::OthelloTrueModel direct;
      r::InferenceRequests req = client_requests(c, enc, direct);
      r::InferenceResults expected, got;
      direct.infer(req, expected);
      for (int i = 0; i < ROUNDS; ++i){
        server.infer(req, got);
        if (not same_results(expected, got)) ++mismatches;
      }
      server.remove_client();
    });
  for (auto& t : clients) t.join();
  server.stop();
  serving.join();

  EXPECT_EQ(mismatches.load(), 0);
  EXPECT_LT(server.num_batches(), (uint64_t)CLIENTS * ROUNDS) << "requests are batched together";
  EXPECT_EQ(server.num_batches(), (uint64_t)backend->calls.load());
}

TEST(InferenceServer, RespectsMaxBatch){
  r::OthelloEncoder enc;
  auto backend = std::make_shared<rt::OthelloTrueModel>();
  r::InferenceServerConfig cfg;
  cfg.max_batch = 4;
  r::InferenceServer server(backend, cfg);
  std::thread serving([&]{ server.run(); });
  rt::OthelloTrueModel direct;
  std::vector<std::thread> clients;
  for (int c = 0; c < 6; ++c){
    server.add_client();
    clients.emplace_back([&, c]{
      r::InferenceRequests req = client_requests(0, enc, direct);  // 3 requests each
      r::InferenceResults res;
      server.infer(req, res);
      server.remove_client();
    });
  }
  for (auto& t : clients) t.join();
  server.stop();
  serving.join();
  // 3 + 3 exceeds 4, so every batch holds a single client
  EXPECT_EQ(server.num_batches(), 6u);
  EXPECT_EQ(server.num_positions(), 18u);
}

TEST(InferenceServer, InstallsWeightsBetweenBatches){
  r::OthelloEncoder enc;
  auto backend = std::make_shared<rt::OthelloTrueModel>();
  r::InferenceServer server(backend, r::InferenceServerConfig{});
  std::thread serving([&]{ server.run(); });
  server.add_client();
  EXPECT_EQ(server.weights_version(), 0u);
  server.set_weights(r::nn::Params{}, 42);
  rt::OthelloTrueModel direct;
  r::InferenceRequests req = client_requests(1, enc, direct);
  r::InferenceResults res;
  server.infer(req, res);
  EXPECT_EQ(server.weights_version(), 42u);
  EXPECT_EQ(backend->weight_updates.load(), 1);
  server.remove_client();
  server.stop();
  serving.join();
}

TEST(InferenceServer, AnswersAfterStop){
  r::OthelloEncoder enc;
  auto backend = std::make_shared<rt::OthelloTrueModel>();
  r::InferenceServer server(backend, r::InferenceServerConfig{});
  server.stop();
  rt::OthelloTrueModel direct;
  r::InferenceRequests req = client_requests(2, enc, direct);
  r::InferenceResults res;
  server.infer(req, res);  // must not block
  EXPECT_EQ((int)res.values.size(), req.size());
  EXPECT_EQ(res.policy_logits.size(), (size_t)req.size() * 65);
}
