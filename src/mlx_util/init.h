#ifndef RLBG_MLX_UTIL_INIT_H
#define RLBG_MLX_UTIL_INIT_H

#include <functional>
#include <string>

#include <mlx/mlx.h>

namespace rlbg::nn::init {

namespace s = std;
namespace mx = mlx::core;

// mlx.nn.init: each factory returns a function mapping an array to a new
// array of the same shape. Python's per-call gain / mode arguments of the
// Glorot and He initializers are factory arguments here. Apply with
// model.apply(init, filter) or by assigning a layer's parameter.
using Initializer = s::function<mx::array(const mx::array&)>;

Initializer constant(double value, mx::Dtype dtype = mx::float32);
Initializer normal(double mean = 0.0, double std = 1.0, mx::Dtype dtype = mx::float32);
Initializer uniform(double low = 0.0, double high = 1.0, mx::Dtype dtype = mx::float32);
Initializer identity(mx::Dtype dtype = mx::float32);
Initializer glorot_normal(mx::Dtype dtype = mx::float32, double gain = 1.0);
Initializer glorot_uniform(mx::Dtype dtype = mx::float32, double gain = 1.0);
// mode is "fan_in" or "fan_out"
Initializer he_normal(
  mx::Dtype dtype = mx::float32,
  const s::string& mode = "fan_in",
  double gain = 1.0
);
Initializer he_uniform(
  mx::Dtype dtype = mx::float32,
  const s::string& mode = "fan_in",
  double gain = 1.0
);
Initializer sparse(
  double sparsity,
  double mean = 0.0,
  double std = 1.0,
  mx::Dtype dtype = mx::float32
);
Initializer orthogonal(double gain = 1.0, mx::Dtype dtype = mx::float32);

} // namespace rlbg::nn::init

#endif//RLBG_MLX_UTIL_INIT_H
