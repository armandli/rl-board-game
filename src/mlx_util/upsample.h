#ifndef RLBG_MLX_UTIL_UPSAMPLE_H
#define RLBG_MLX_UTIL_UPSAMPLE_H

#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// scale_factor holds one value for every spatial axis, or a single value
// for all of them
mx::array upsample_nearest(const mx::array& x, const s::vector<double>& scale_factor);
mx::array upsample_linear(
  const mx::array& x,
  const s::vector<double>& scale_factor,
  bool align_corners = false,
  bool antialias = false
);
mx::array upsample_cubic(
  const mx::array& x,
  const s::vector<double>& scale_factor,
  bool align_corners = false,
  bool antialias = false
);

// upsamples the spatial axes of channels-last (N, *spatial, C) inputs;
// mode is "nearest", "linear" or "cubic"
struct Upsample : UnaryModule {
  Upsample(
    s::vector<double> scale_factor,
    s::string mode = "nearest",
    bool align_corners = false,
    bool antialias = false
  );
  Upsample(
    double scale_factor,
    s::string mode = "nearest",
    bool align_corners = false,
    bool antialias = false)
    : Upsample(s::vector<double>{scale_factor}, s::move(mode), align_corners, antialias)
  {
    mScalar = true;
  }

  s::string type_name() const override { return "Upsample"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  s::vector<double> scale_factor;
  s::string mode;
  bool align_corners;
  bool antialias;

private:
  bool mScalar = false;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_UPSAMPLE_H
