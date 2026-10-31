#ifndef RLBG_MODEL_TRANSFORMER_MUZERO_H
#define RLBG_MODEL_TRANSFORMER_MUZERO_H

#include <memory>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include <model/muzero_model.h>

namespace rlbg {

namespace s = std;
namespace mx = mlx::core;

struct TransformerMuZeroConfig {
  // set from the encoder
  int num_tokens = 64;
  int num_features = 4;
  int num_actions = 65;
  int num_token_classes = 3;

  int dims = 128;
  int num_heads = 4;
  int repr_layers = 6;
  int dyn_layers = 4;
  int pred_layers = 1;
  int proj_dims = 128;
};

// MuZero with every network a bidirectional transformer over board tokens
// instead of a ResNet over board planes; attention reaches across the board
// in one layer, which suits othello's long range flips.
//
//   represent: linear token embedding + learned position embedding,
//     repr_layers pre-norm blocks, LayerNorm without affine parameters
//     (bounds the latent like MuZero's min-max scaling of hidden states)
//   dynamics: a learned action token is prepended and the action's square is
//     marked, dyn_layers blocks, the action token reads out the reward, the
//     board tokens give the next latent after the same normalization
//   predict: pred_layers blocks; per-square policy logits plus a pooled pass
//     logit (when num_actions == num_tokens + 1, else a pooled linear head),
//     pooled MLP value head, per-token ownership head
struct TransformerMuZero : MuZeroModel {
  explicit TransformerMuZero(const TransformerMuZeroConfig& config);

  s::string type_name() const override { return "TransformerMuZero"; }

  mx::array represent(const mx::array& features) override;
  DynamicsOutput dynamics(const mx::array& latent, const mx::array& actions) override;
  Prediction predict(const mx::array& latent) override;
  mx::array project(const mx::array& latent) override;
  mx::array predict_projection(const mx::array& projection) override;
  mx::array reconstruct(const mx::array& latent) override;

  int num_tokens() const override { return config.num_tokens; }
  int latent_dims() const override { return config.dims; }
  int num_actions() const override { return config.num_actions; }

  TransformerMuZeroConfig config;

  s::shared_ptr<nn::Linear> input_proj;
  mx::array pos_embedding;
  s::shared_ptr<nn::ModuleList> repr_blocks;
  s::shared_ptr<nn::LayerNorm> repr_norm;

  s::shared_ptr<nn::Embedding> action_embedding;
  mx::array action_marker;
  s::shared_ptr<nn::ModuleList> dyn_blocks;
  s::shared_ptr<nn::LayerNorm> dyn_norm;
  s::shared_ptr<nn::RMSNorm> reward_norm;
  s::shared_ptr<nn::Linear> reward_head;

  s::shared_ptr<nn::ModuleList> pred_blocks;
  s::shared_ptr<nn::RMSNorm> pred_norm;
  s::shared_ptr<nn::Linear> policy_head;
  s::shared_ptr<nn::Linear> pass_head;
  s::shared_ptr<nn::Sequential> value_head;
  s::shared_ptr<nn::Linear> ownership_head;

  s::shared_ptr<nn::Linear> recon_head;
  s::shared_ptr<nn::Sequential> projector;
  s::shared_ptr<nn::Sequential> predictor;

protected:
  static mx::array run_blocks(const nn::ModuleList& blocks, mx::array x);
  bool per_square_policy() const { return config.num_actions == config.num_tokens + 1; }
};

}  // namespace rlbg

#endif//RLBG_MODEL_TRANSFORMER_MUZERO_H
