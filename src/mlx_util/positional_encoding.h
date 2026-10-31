#ifndef RLBG_MLX_UTIL_POSITIONAL_ENCODING_H
#define RLBG_MLX_UTIL_POSITIONAL_ENCODING_H

#include <optional>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// rotary positional encoding over the first dims features of the last axis
struct RoPE : Module {
  explicit RoPE(int dims, bool traditional = false, float base = 10000.0f, float scale = 1.0f)
    : dims(dims), traditional(traditional), base(base), scale(scale) {}

  s::string type_name() const override { return "RoPE"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x, int offset = 0) const {
    return mx::fast::rope(x, dims, traditional, base, scale, offset);
  }
  mx::array operator()(const mx::array& x, int offset = 0) const {
    return forward(x, offset);
  }

  int dims;
  bool traditional;
  float base;
  float scale;
};

// maps positions x of any shape to (*x.shape, dims) features
struct SinusoidalPositionalEncoding : UnaryModule {
  explicit SinusoidalPositionalEncoding(
    int dims,
    double min_freq = 0.0001,
    double max_freq = 1.0,
    s::optional<double> scale = s::nullopt,
    bool cos_first = false,
    bool full_turns = false
  );

  s::string type_name() const override { return "SinusoidalPositionalEncoding"; }
  mx::array forward(const mx::array& x) override;

  double scale;
  bool cos_first;

private:
  // not a parameter: Python keeps it in the private _sigmas attribute
  mx::array mSigmas;
};

// attention with linear biases; adds a distance penalty to attention scores
// of shape (N, heads, queries, keys)
struct ALiBi : Module {
  s::string type_name() const override { return "ALiBi"; }

  static mx::array create_alibi_matrix(
    int q_sequence_length,
    int k_sequence_length,
    int num_heads,
    int offset = 0,
    mx::Dtype dtype = mx::float32
  );
  static mx::array create_alibi_slope(int num_heads, mx::Dtype dtype = mx::float32);

  mx::array forward(
    const mx::array& attention_scores,
    int offset = 0,
    const s::optional<mx::array>& mask = s::nullopt
  ) const;
  mx::array operator()(
    const mx::array& attention_scores,
    int offset = 0,
    const s::optional<mx::array>& mask = s::nullopt) const
  {
    return forward(attention_scores, offset, mask);
  }
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_POSITIONAL_ENCODING_H
