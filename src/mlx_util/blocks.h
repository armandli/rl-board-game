#ifndef RLBG_MLX_UTIL_BLOCKS_H
#define RLBG_MLX_UTIL_BLOCKS_H

#include <memory>
#include <optional>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/linear.h>
#include <mlx_util/module.h>
#include <mlx_util/normalization.h>
#include <mlx_util/transformer.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// building blocks of modern (LLaMA style) transformers that mlx.nn does not
// ship as layers

// gated MLP: w2(silu(w1 x) * w3 x)
struct SwiGLU : UnaryModule {
  SwiGLU(int dims, int hidden_dims, bool bias = false);

  s::string type_name() const override { return "SwiGLU"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  s::shared_ptr<Linear> w1;
  s::shared_ptr<Linear> w2;
  s::shared_ptr<Linear> w3;
};

// hidden size with about the parameters of a 4x GELU MLP: 8/3 dims rounded
// up to a multiple of 32
int swiglu_hidden_dims(int dims);

// pre-norm residual block: x + attn(rms(x)), then x + swiglu(rms(x));
// unset mlp_dims means swiglu_hidden_dims(dims)
struct TransformerBlock : Module {
  TransformerBlock(int dims, int num_heads, s::optional<int> mlp_dims = s::nullopt);

  s::string type_name() const override { return "TransformerBlock"; }
  mx::array forward(const mx::array& x, const Mask& mask = s::nullopt);
  mx::array operator()(const mx::array& x, const Mask& mask = s::nullopt) {
    return forward(x, mask);
  }

  s::shared_ptr<RMSNorm> attention_norm;
  s::shared_ptr<MultiHeadAttention> attention;
  s::shared_ptr<RMSNorm> mlp_norm;
  s::shared_ptr<SwiGLU> mlp;
};

// x * scale in the forward pass, gradient scaled by scale in the backward
// pass (MuZero's scale_gradient)
inline mx::array scale_gradient(const mx::array& x, double scale) {
  return x * scalar_like(scale, x) + mx::stop_gradient(x) * scalar_like(1.0 - scale, x);
}

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_BLOCKS_H
