#ifndef RLBG_MLX_UTIL_POOLING_H
#define RLBG_MLX_UTIL_POOLING_H

#include <cmath>

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/int_tuple.h>
#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// pools channels-last inputs (N, *spatial, C) over sliding windows
struct Pool : UnaryModule {
  using Reduce = s::function<mx::array(const mx::array&, const s::vector<int>&)>;

  Pool(
    s::string name,
    Reduce reduce,
    float padding_value,
    s::vector<int> kernel_size,
    s::vector<int> stride,
    s::vector<int> padding
  );

  s::string type_name() const override { return mName; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  s::vector<int> kernel_size;
  s::vector<int> stride;
  s::vector<int> padding;

private:
  s::string mName;
  Reduce mReduce;
  float mPaddingValue;
  s::vector<int> mAxes;
};

// stride defaults to the kernel size as in Python
template <size_t N>
struct PoolNd : Pool {
  PoolNd(
    s::string name,
    Reduce reduce,
    float padding_value,
    IntN<N> kernel_size,
    IntN<N> stride,
    IntN<N> padding)
    : Pool(
        s::move(name),
        s::move(reduce),
        padding_value,
        kernel_size.vector(),
        stride.given ? stride.vector() : kernel_size.vector(),
        padding.vector()
      ) {}
};

mx::array max_reduce(const mx::array& x, const s::vector<int>& axes);
mx::array mean_reduce(const mx::array& x, const s::vector<int>& axes);
// as mlx.nn's sliding window helper: (N, *out, *window, C)
mx::array sliding_windows(
  const mx::array& x,
  const s::vector<int>& window_shape,
  const s::vector<int>& window_strides
);

#define RLBG_NN_POOL(NAME, DIM, REDUCE, PAD)                               \
  struct NAME : PoolNd<DIM> {                                              \
    explicit NAME(                                                         \
      IntN<DIM> kernel_size,                                               \
      IntN<DIM> stride = IntN<DIM>(),                                      \
      IntN<DIM> padding = 0)                                               \
      : PoolNd<DIM>(#NAME, REDUCE, PAD, kernel_size, stride, padding) {}   \
  };

RLBG_NN_POOL(MaxPool1d, 1, max_reduce, -INFINITY)
RLBG_NN_POOL(AvgPool1d, 1, mean_reduce, 0.0f)
RLBG_NN_POOL(MaxPool2d, 2, max_reduce, -INFINITY)
RLBG_NN_POOL(AvgPool2d, 2, mean_reduce, 0.0f)
RLBG_NN_POOL(MaxPool3d, 3, max_reduce, -INFINITY)
RLBG_NN_POOL(AvgPool3d, 3, mean_reduce, 0.0f)

#undef RLBG_NN_POOL

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_POOLING_H
