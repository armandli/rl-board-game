#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <model/model_inference.h>
#include <model/transformer_muzero.h>

#include <common/othello_fixtures.h>

namespace mx = mlx::core;
namespace r = rlbg;
namespace rt = rlbg::test;

namespace {

struct Requests {
  r::InferenceRequests initial, recurrent, mixed;
};

// observations of positions along a random game, plus recurrent requests
// on the latents of those observations
Requests build_requests(const r::OthelloEncoder& enc, r::Inference& inf, int n){
  Requests q;
  for (r::InferenceRequests* rq : {&q.initial, &q.recurrent, &q.mixed}){
    rq->observation_size = enc.observation_size();
    rq->latent_size = inf.latent_size();
  }
  std::mt19937_64 rng(1);
  r::OthelloBitboard g;
  std::vector<uint8_t> obs(enc.observation_size());
  r::ActionList<65> legal;
  for (int i = 0; i < n; ++i){
    enc.encode(g, obs.data());
    q.initial.add_initial(obs.data());
    q.mixed.add_initial(obs.data());
    g.legal_actions(legal);
    g.apply(legal[rng() % legal.size]);
  }
  r::InferenceResults res;
  inf.infer(q.initial, res);
  for (int i = 0; i < n; ++i){
    q.recurrent.add_recurrent(res.latent(i), i % 65);
    q.mixed.add_recurrent(res.latent(i), i % 65);
  }
  return q;
}

float max_diff(const std::vector<float>& a, const std::vector<float>& b, size_t a0, size_t b0, size_t n){
  float d = 0.f;
  for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs(a[a0 + i] - b[b0 + i]));
  return d;
}

} // namespace

TEST(ModelInference, ParsesDtypes){
  EXPECT_EQ(r::parse_dtype("float32"), mx::float32);
  EXPECT_EQ(r::parse_dtype("float16"), mx::float16);
  EXPECT_EQ(r::parse_dtype("bfloat16"), mx::bfloat16);
  EXPECT_THROW(r::parse_dtype("int8"), std::invalid_argument);
}

TEST(ModelInference, ResultLayout){
  r::OthelloEncoder enc;
  r::ModelInference inf(rt::small_model(enc), enc.observation_size());
  Requests q = build_requests(enc, inf, 5);
  r::InferenceResults res;
  inf.infer(q.mixed, res);
  EXPECT_EQ(res.num_initial, 5);
  EXPECT_EQ(res.values.size(), 10u);
  EXPECT_EQ(res.policy_logits.size(), 10u * 65);
  EXPECT_EQ(res.latents.size(), 10u * inf.latent_size());
  for (int i = 0; i < 10; ++i){
    EXPECT_LE(std::abs(res.values[i]), 1.f);
    EXPECT_LE(std::abs(res.rewards[i]), 1.f);
  }
  for (int i = 0; i < 5; ++i) EXPECT_EQ(res.rewards[res.initial_index(i)], 0.f);
}

// a batch mixing both kinds of requests gives what separate batches give
TEST(ModelInference, MixedBatchMatchesSeparateBatches){
  r::OthelloEncoder enc;
  r::ModelInference inf(rt::small_model(enc), enc.observation_size());
  Requests q = build_requests(enc, inf, 6);
  r::InferenceResults ri, rr, rm;
  inf.infer(q.initial, ri);
  inf.infer(q.recurrent, rr);
  inf.infer(q.mixed, rm);
  EXPECT_LT(max_diff(rm.policy_logits, ri.policy_logits, 0, 0, 6 * 65), 1e-4f);
  EXPECT_LT(max_diff(rm.policy_logits, rr.policy_logits, 6 * 65, 0, 6 * 65), 1e-4f);
  EXPECT_LT(max_diff(rm.values, rr.values, 6, 0, 6), 1e-4f);
  EXPECT_LT(max_diff(rm.rewards, rr.rewards, 6, 0, 6), 1e-4f);
}

TEST(ModelInference, HalfPrecisionMatchesFloat){
  r::OthelloEncoder enc;
  auto m32 = rt::small_model(enc);
  auto m16 = rt::small_model(enc);
  m16->update(m32->parameters());
  r::ModelInference f32(m32, enc.observation_size(), mx::float32);
  r::ModelInference f16(m16, enc.observation_size(), mx::float16);
  Requests q = build_requests(enc, f32, 8);
  r::InferenceResults a, b;
  f32.infer(q.mixed, a);
  f16.infer(q.mixed, b);
  EXPECT_LT(max_diff(a.policy_logits, b.policy_logits, 0, 0, a.policy_logits.size()), 0.05f);
  EXPECT_LT(max_diff(a.values, b.values, 0, 0, a.values.size()), 0.02f);
}

TEST(ModelInference, SetWeightsReplacesTheNetwork){
  r::OthelloEncoder enc;
  auto other = rt::small_model(enc);
  r::ModelInference inf(rt::small_model(enc), enc.observation_size(), mx::float16);
  r::ModelInference ref(other, enc.observation_size(), mx::float16);
  Requests q = build_requests(enc, inf, 4);
  r::InferenceResults before, after, expected;
  inf.infer(q.initial, before);
  inf.set_weights(other->parameters());
  inf.infer(q.initial, after);
  ref.infer(q.initial, expected);
  EXPECT_GT(max_diff(before.policy_logits, after.policy_logits, 0, 0, 4 * 65), 1e-3f);
  EXPECT_LT(max_diff(after.policy_logits, expected.policy_logits, 0, 0, 4 * 65), 1e-3f);
  // the model keeps the inference precision after new float32 weights
  EXPECT_EQ(inf.model().parameters().at("input_proj.weight").dtype(), mx::float16);
}

// regression: MLX binds lazy arrays to the creating thread's stream, so a
// model built on one thread must be usable from another
TEST(ModelInference, UsableFromAnotherThread){
  r::OthelloEncoder enc;
  auto inf = std::make_shared<r::ModelInference>(rt::small_model(enc), enc.observation_size(), mx::float16);
  Requests q = build_requests(enc, *inf, 3);
  r::InferenceResults here, there;
  inf->infer(q.mixed, here);
  std::thread t([&]{ inf->infer(q.mixed, there); });
  t.join();
  EXPECT_LT(max_diff(here.values, there.values, 0, 0, here.values.size()), 1e-3f);
}
