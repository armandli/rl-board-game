#ifndef RLBG_LIBTORCH_TRANSFORMER_MUZERO_H
#define RLBG_LIBTORCH_TRANSFORMER_MUZERO_H

#include <cstdint>

#include <string>
#include <vector>

#include <torch/torch.h>

namespace rlbg::libtorch {

namespace s = std;
namespace t = torch;

// value and reward heads predict 3 outcome classes from the mover's point of
// view (loss, draw, win, i.e. -1, 0, +1); the scalar is P(win) - P(loss)
constexpr int NUM_OUTCOME_CLASSES = 3;

inline t::Tensor outcome_scalar(const t::Tensor& logits){
  t::Tensor p = t::softmax(logits.to(t::kFloat), -1);
  return p.select(-1, 2) - p.select(-1, 0);
}

struct TransformerMuZeroConfig {
  // set from the encoder
  int64_t num_tokens = 64;
  int64_t num_features = 4;
  int64_t num_actions = 65;
  int64_t num_token_classes = 3;

  int64_t dims = 256;
  int64_t num_heads = 8;
  int64_t repr_layers = 6;
  int64_t dyn_layers = 4;
  int64_t pred_layers = 2;
  int64_t proj_dims = 256;

  s::vector<int64_t> to_vector() const;
  static TransformerMuZeroConfig from_vector(const s::vector<int64_t>& v);
  bool operator==(const TransformerMuZeroConfig&) const = default;
  s::string describe() const;
};

struct RMSNormImpl : t::nn::Module {
  explicit RMSNormImpl(int64_t dims, double eps = 1e-6);
  t::Tensor forward(const t::Tensor& x);

  t::Tensor weight;
  double    eps;
};
TORCH_MODULE(RMSNorm);

// bidirectional multi-head self-attention with PyTorch's fused kernel (flash
// or memory efficient attention in half precision)
struct AttentionImpl : t::nn::Module {
  AttentionImpl(int64_t dims, int64_t heads);
  t::Tensor forward(const t::Tensor& x);

  int64_t       heads;
  t::nn::Linear qkv{nullptr}, out{nullptr};
};
TORCH_MODULE(Attention);

// gated MLP with 8/3 the width, as many parameters as a 4x MLP
struct SwiGLUImpl : t::nn::Module {
  explicit SwiGLUImpl(int64_t dims);
  t::Tensor forward(const t::Tensor& x);

  t::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};
};
TORCH_MODULE(SwiGLU);

// pre-norm transformer block
struct BlockImpl : t::nn::Module {
  BlockImpl(int64_t dims, int64_t heads);
  t::Tensor forward(t::Tensor x);

  RMSNorm   norm1{nullptr}, norm2{nullptr};
  Attention attn{nullptr};
  SwiGLU    mlp{nullptr};
};
TORCH_MODULE(Block);

struct Prediction {
  t::Tensor policy_logits;    // (B, A)
  t::Tensor value_logits;     // (B, 3)
  t::Tensor ownership_logits; // (B, T, C) token classes at the end of the game
};

struct DynamicsOutput {
  t::Tensor latent;           // (B, T, D)
  t::Tensor reward_logits;    // (B, 3) reward of the action for its mover
};

// MuZero with every network a bidirectional transformer over board tokens,
// one token per square; attention reaches across the board in one layer,
// which suits othello's long range flips.
//
//   represent h: the game state embedding of every token is a learned
//     embedding of its class (the encoder's first num_token_classes features
//     one-hot encode it: empty, own, opponent), a projection of its other
//     features (legal placement) and a learned position embedding;
//     repr_layers blocks, then LayerNorm without affine parameters, which
//     bounds the latent like MuZero's min-max scaling of hidden states
//   dynamics g: a learned action token is prepended and the action's square
//     is marked, dyn_layers blocks; the action token reads out the reward,
//     the board tokens give the next latent after the same normalization
//   predict f: pred_layers blocks; per-square policy logits plus a pooled
//     pass logit, pooled MLP value head, per-token ownership head
//   project / predict_projection: SimSiam heads (EfficientZero consistency)
//   reconstruct: token classes of the board the latent stands for
struct TransformerMuZeroImpl : t::nn::Module {
  explicit TransformerMuZeroImpl(const TransformerMuZeroConfig& config);

  // features (B, T, F) uint8 -> latent (B, T, D)
  t::Tensor represent(const t::Tensor& features);
  // latent (B, T, D), actions (B,) int64
  DynamicsOutput dynamics(const t::Tensor& latent, const t::Tensor& actions);
  Prediction predict(const t::Tensor& latent);
  t::Tensor project(const t::Tensor& latent);
  t::Tensor predict_projection(const t::Tensor& projection);
  // (B, T, C) logits
  t::Tensor reconstruct(const t::Tensor& latent);

  int64_t num_parameters() const;
  int64_t latent_size() const { return config.num_tokens * config.dims; }

  TransformerMuZeroConfig config;

  t::nn::Embedding  class_embedding{nullptr};
  t::nn::Linear     feature_proj{nullptr};
  t::Tensor         pos_embedding;
  t::nn::ModuleList repr_blocks{nullptr};
  t::nn::LayerNorm  repr_norm{nullptr};

  t::nn::Embedding  action_embedding{nullptr};
  t::Tensor         action_marker;
  t::nn::ModuleList dyn_blocks{nullptr};
  t::nn::LayerNorm  dyn_norm{nullptr};
  RMSNorm           reward_norm{nullptr};
  t::nn::Linear     reward_head{nullptr};

  t::nn::ModuleList pred_blocks{nullptr};
  RMSNorm           pred_norm{nullptr};
  t::nn::Linear     policy_head{nullptr}, pass_head{nullptr};
  t::nn::Sequential value_head{nullptr};
  t::nn::Linear     ownership_head{nullptr};

  t::nn::Linear     recon_head{nullptr};
  t::nn::Sequential projector{nullptr}, predictor{nullptr};

protected:
  t::nn::ModuleList make_blocks(int64_t layers);
  static t::Tensor run_blocks(t::nn::ModuleList& blocks, t::Tensor x);
};
TORCH_MODULE(TransformerMuZero);

// copies every parameter and buffer of src into dst; dtypes and devices may
// differ, configurations must match
void copy_weights(TransformerMuZero& dst, TransformerMuZero& src);

// a model file holds its configuration followed by the weights, so it loads
// without knowing the model's size
void save_model(TransformerMuZero& model, const s::string& path);
TransformerMuZero load_model(const s::string& path, t::Device device = t::kCPU);
// the configuration stored in a model file
TransformerMuZeroConfig load_model_config(const s::string& path);

}  // namespace rlbg::libtorch

#endif//RLBG_LIBTORCH_TRANSFORMER_MUZERO_H
