#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <encoder/othello_encoder.h>
#include <model/transformer_muzero.h>

#include <common/othello_fixtures.h>
#include <common/temp_dir.h>

namespace mx = mlx::core;
namespace r = rlbg;
namespace rt = rlbg::test;

namespace {

mx::array random_features(int b){
  return mx::astype(mx::random::bernoulli(0.3f, {b, 64, 4}), mx::float32);
}

bool all_close(const mx::array& a, const mx::array& b, float tol = 1e-5f){
  return a.shape() == b.shape() and mx::allclose(a, b, tol, tol).item<bool>();
}

} // namespace

TEST(TransformerMuZero, OutputShapes){
  r::OthelloEncoder enc;
  auto m = rt::small_model(enc);
  mx::array lat = m->represent(random_features(3));
  EXPECT_EQ(lat.shape(), (mx::Shape{3, 64, 32}));
  EXPECT_EQ(m->latent_size(), 64 * 32);
  r::DynamicsOutput dyn = m->dynamics(lat, mx::array({0, 20, 64}));
  EXPECT_EQ(dyn.latent.shape(), lat.shape());
  EXPECT_EQ(dyn.reward_logits.shape(), (mx::Shape{3, 3}));
  r::Prediction p = m->predict(lat);
  EXPECT_EQ(p.policy_logits.shape(), (mx::Shape{3, 65}));
  EXPECT_EQ(p.value_logits.shape(), (mx::Shape{3, 3}));
  EXPECT_EQ(p.ownership_logits.shape(), (mx::Shape{3, 64, 3}));
  EXPECT_EQ(m->reconstruct(lat).shape(), (mx::Shape{3, 64, 3}));
  mx::array proj = m->project(lat);
  EXPECT_EQ(proj.shape(), (mx::Shape{3, 64, 32}));
  EXPECT_EQ(m->predict_projection(proj).shape(), proj.shape());
}

TEST(TransformerMuZero, LatentsAreNormalizedPerToken){
  r::OthelloEncoder enc;
  auto m = rt::small_model(enc);
  mx::array lat = m->represent(random_features(4));
  r::DynamicsOutput dyn = m->dynamics(lat, mx::array({1, 2, 3, 64}));
  for (const mx::array& x : {lat, dyn.latent}){
    EXPECT_LT(mx::max(mx::abs(mx::mean(x, -1))).item<float>(), 1e-4f);
    EXPECT_LT(mx::max(mx::abs(mx::var(x, -1) - mx::array(1.f))).item<float>(), 1e-2f);
  }
}

TEST(TransformerMuZero, DynamicsDependsOnTheAction){
  r::OthelloEncoder enc;
  auto m = rt::small_model(enc);
  mx::array lat = m->represent(random_features(1));
  mx::array a = m->dynamics(lat, mx::array({10})).latent;
  mx::array b = m->dynamics(lat, mx::array({11})).latent;
  mx::array pass = m->dynamics(lat, mx::array({64})).latent;
  EXPECT_FALSE(all_close(a, b));
  EXPECT_FALSE(all_close(a, pass));
  EXPECT_TRUE(all_close(a, m->dynamics(lat, mx::array({10})).latent));
}

TEST(TransformerMuZero, PooledPolicyHeadForOtherActionSpaces){
  r::TransformerMuZeroConfig c;
  c.num_tokens = 9;
  c.num_features = 3;
  c.num_actions = 4;   // not one per token plus pass
  c.dims = 16;
  c.num_heads = 2;
  c.repr_layers = c.dyn_layers = c.pred_layers = 1;
  c.proj_dims = 16;
  r::TransformerMuZero m(c);
  EXPECT_FALSE(m.parameters().contains("pass_head.weight"));
  mx::array lat = m.represent(mx::zeros({2, 9, 3}));
  EXPECT_EQ(m.predict(lat).policy_logits.shape(), (mx::Shape{2, 4}));
}

TEST(TransformerMuZero, ParameterGroupsAndCount){
  r::OthelloEncoder enc;
  auto m = rt::small_model(enc);
  r::nn::Params p = m->parameters();
  for (const char* key : {"input_proj.weight", "pos_embedding", "repr_blocks.0.mlp.w1.weight",
                          "action_embedding.weight", "action_marker", "dyn_blocks.0.attention.query_proj.weight",
                          "reward_head.weight", "policy_head.weight", "pass_head.weight",
                          "value_head.layers.2.weight", "ownership_head.weight", "recon_head.weight",
                          "projector.layers.0.weight", "predictor.layers.3.weight"})
    EXPECT_TRUE(p.contains(key)) << key;
  size_t n = 0;
  for (const auto& [k, v] : p) n += v.size();
  EXPECT_EQ(m->num_parameters(), n);
  // the default model is sized for strong othello on one machine
  r::TransformerMuZeroConfig def;
  EXPECT_NEAR(r::TransformerMuZero(def).num_parameters() / 1e6, 2.31, 0.05);
}

TEST(TransformerMuZero, RejectsHeadsThatDoNotDivideDims){
  r::TransformerMuZeroConfig c;
  c.dims = 30;
  c.num_heads = 4;
  EXPECT_THROW(r::TransformerMuZero{c}, std::invalid_argument);
}

TEST(TransformerMuZero, SaveAndLoadWeights){
  rt::TempDir dir;
  r::OthelloEncoder enc;
  auto a = rt::small_model(enc);
  auto b = rt::small_model(enc);
  mx::array x = random_features(2);
  EXPECT_FALSE(all_close(a->represent(x), b->represent(x)));
  std::string file = (dir / "m.safetensors").string();
  a->save_weights(file);
  b->load_weights(file);
  EXPECT_TRUE(all_close(a->represent(x), b->represent(x)));
  EXPECT_TRUE(all_close(a->predict(a->represent(x)).policy_logits, b->predict(b->represent(x)).policy_logits));
}

TEST(OutcomeHelpers, ScalarAndClasses){
  EXPECT_EQ(r::outcome_class(-1), 0);
  EXPECT_EQ(r::outcome_class(0), 1);
  EXPECT_EQ(r::outcome_class(1), 2);
  mx::array logits = mx::array({0.f, 0.f, 100.f, 100.f, 0.f, 0.f, 0.f, 0.f, 0.f}, {3, 3});
  mx::array v = r::outcome_scalar(logits);
  EXPECT_TRUE(all_close(v, mx::array({1.f, -1.f, 0.f}), 1e-4f));
}
