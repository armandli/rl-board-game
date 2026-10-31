#ifndef RLBG_MLX_UTIL_LINEAR_H
#define RLBG_MLX_UTIL_LINEAR_H

#include <optional>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

struct Identity : UnaryModule {
  s::string type_name() const override { return "Identity"; }
  mx::array forward(const mx::array& x) override { return x; }
};

// weight is (output_dims, input_dims) as in Python
struct Linear : UnaryModule {
  Linear(int input_dims, int output_dims, bool bias = true);

  s::string type_name() const override { return "Linear"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
};

// weight is (output_dims, input2_dims, input1_dims) as in Python
struct Bilinear : Module {
  Bilinear(int input1_dims, int input2_dims, int output_dims, bool bias = true);

  s::string type_name() const override { return "Bilinear"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x1, const mx::array& x2);
  mx::array operator()(const mx::array& x1, const mx::array& x2) {
    return forward(x1, x2);
  }

  mx::array weight;
  s::optional<mx::array> bias;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_LINEAR_H
