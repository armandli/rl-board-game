#ifndef RLBG_MLX_UTIL_CONVOLUTION_H
#define RLBG_MLX_UTIL_CONVOLUTION_H

#include <optional>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/int_tuple.h>
#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// MLX convolutions are channels last: inputs are (N, L, C), (N, H, W, C) or
// (N, D, H, W, C) and weights are (out, *kernel, in / groups)

struct Conv1d : UnaryModule {
  Conv1d(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
    bool bias = true
  );

  s::string type_name() const override { return "Conv1d"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
  int padding;
  int dilation;
  int stride;
  int groups;
};

struct Conv2d : UnaryModule {
  Conv2d(
    int in_channels,
    int out_channels,
    Int2 kernel_size,
    Int2 stride = 1,
    Int2 padding = 0,
    Int2 dilation = 1,
    int groups = 1,
    bool bias = true
  );

  s::string type_name() const override { return "Conv2d"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
  Int2 padding;
  Int2 stride;
  Int2 dilation;
  int groups;
};

struct Conv3d : UnaryModule {
  Conv3d(
    int in_channels,
    int out_channels,
    Int3 kernel_size,
    Int3 stride = 1,
    Int3 padding = 0,
    Int3 dilation = 1,
    bool bias = true
  );

  s::string type_name() const override { return "Conv3d"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
  Int3 padding;
  Int3 stride;
  Int3 dilation;
};

struct ConvTranspose1d : UnaryModule {
  ConvTranspose1d(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int output_padding = 0,
    bool bias = true
  );

  s::string type_name() const override { return "ConvTranspose1d"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
  int padding;
  int dilation;
  int stride;
  int output_padding;
};

struct ConvTranspose2d : UnaryModule {
  ConvTranspose2d(
    int in_channels,
    int out_channels,
    Int2 kernel_size,
    Int2 stride = 1,
    Int2 padding = 0,
    Int2 dilation = 1,
    Int2 output_padding = 0,
    bool bias = true
  );

  s::string type_name() const override { return "ConvTranspose2d"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
  Int2 padding;
  Int2 stride;
  Int2 dilation;
  Int2 output_padding;
};

struct ConvTranspose3d : UnaryModule {
  ConvTranspose3d(
    int in_channels,
    int out_channels,
    Int3 kernel_size,
    Int3 stride = 1,
    Int3 padding = 0,
    Int3 dilation = 1,
    Int3 output_padding = 0,
    bool bias = true
  );

  s::string type_name() const override { return "ConvTranspose3d"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  mx::array weight;
  s::optional<mx::array> bias;
  Int3 padding;
  Int3 stride;
  Int3 dilation;
  Int3 output_padding;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_CONVOLUTION_H
