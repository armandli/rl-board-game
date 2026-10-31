#ifndef RLBG_MLX_UTIL_DROPOUT_H
#define RLBG_MLX_UTIL_DROPOUT_H

#include <string>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// zeroes elements with probability p while training; identity in eval()
struct Dropout : UnaryModule {
  explicit Dropout(double p = 0.5);

  s::string type_name() const override { return "Dropout"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

protected:
  // number of trailing-but-last axes the mask is shared across; 0 drops
  // single elements, 2 drops whole channels of an NHWC input
  mx::array apply_mask(const mx::array& x, int shared_axes) const;

  double mP1;
};

// drops whole channels of (N, H, W, C) or (H, W, C) inputs
struct Dropout2d : Dropout {
  explicit Dropout2d(double p = 0.5) : Dropout(p) {}

  s::string type_name() const override { return "Dropout2d"; }
  mx::array forward(const mx::array& x) override;
};

// drops whole channels of (N, D, H, W, C) or (D, H, W, C) inputs
struct Dropout3d : Dropout {
  explicit Dropout3d(double p = 0.5) : Dropout(p) {}

  s::string type_name() const override { return "Dropout3d"; }
  mx::array forward(const mx::array& x) override;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_DROPOUT_H
