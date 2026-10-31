#include <mlx_util/pooling.h>

#include <cmath>

#include <algorithm>
#include <format>
#include <stdexcept>

#include <mlx_util/repr.h>

namespace rlbg::nn {

namespace {

mx::array non_overlapping_sliding_windows(
  const mx::array& x,
  const s::vector<int>& window_shape)
{
  const mx::Shape& shape = x.shape();
  mx::Shape new_shape{shape[0]};
  for (size_t i = 0; i < window_shape.size(); ++i) {
    new_shape.push_back(shape[i + 1] / window_shape[i]);
    new_shape.push_back(window_shape[i]);
  }
  new_shape.push_back(shape.back());
  int last_axis = (int)new_shape.size() - 1;
  s::vector<int> axis_order{0};
  for (int i = 1; i < last_axis; i += 2) axis_order.push_back(i);
  for (int i = 2; i < last_axis; i += 2) axis_order.push_back(i);
  axis_order.push_back(last_axis);
  return mx::transpose(mx::reshape(x, new_shape), axis_order);
}

} // namespace

mx::array max_reduce(const mx::array& x, const s::vector<int>& axes) {
  return mx::max(x, axes);
}

mx::array mean_reduce(const mx::array& x, const s::vector<int>& axes) {
  return mx::mean(x, axes);
}

mx::array sliding_windows(
  const mx::array& x,
  const s::vector<int>& window_shape,
  const s::vector<int>& window_strides)
{
  if (x.ndim() < 3)
    throw s::invalid_argument(s::format(
      "To extract sliding windows at least 1 spatial dimension (3 total) is "
      "needed but the input only has {} dimensions.", x.ndim()
    ));
  size_t spatial = x.ndim() - 2;
  if (spatial != window_shape.size() or spatial != window_strides.size())
    throw s::invalid_argument(s::format(
      "To extract sliding windows the window shapes and strides must have the "
      "same number of spatial dimensions as the signal but the signal has {} "
      "dims and the window shape has {} and strides have {}.",
      spatial, window_shape.size(), window_strides.size()
    ));

  const mx::Shape& shape = x.shape();
  bool non_overlapping = true;
  for (size_t i = 0; i < spatial; ++i)
    if (window_shape[i] != window_strides[i] or shape[i + 1] % window_shape[i] != 0)
      non_overlapping = false;
  if (non_overlapping) return non_overlapping_sliding_windows(x, window_shape);

  // row-major element strides of x
  mx::Strides strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];

  mx::Shape final_shape{shape[0]};
  for (size_t i = 0; i < spatial; ++i)
    final_shape.push_back((shape[i + 1] - window_shape[i]) / window_strides[i] + 1);
  final_shape.insert(final_shape.end(), window_shape.begin(), window_shape.end());
  final_shape.push_back(shape.back());

  mx::Strides final_strides{strides[0]};
  for (size_t i = 0; i < spatial; ++i)
    final_strides.push_back(strides[i + 1] * window_strides[i]);
  for (size_t i = 0; i < spatial; ++i) final_strides.push_back(strides[i + 1]);
  final_strides.push_back(strides.back());

  return mx::as_strided(x, final_shape, final_strides, 0);
}

Pool::Pool(
  s::string name,
  Reduce reduce,
  float padding_value,
  s::vector<int> kernel_size,
  s::vector<int> stride,
  s::vector<int> padding)
  : kernel_size(s::move(kernel_size)),
    stride(s::move(stride)),
    padding(s::move(padding)),
    mName(s::move(name)),
    mReduce(s::move(reduce)),
    mPaddingValue(padding_value)
{
  auto check_positive = [&](const char* what, const s::vector<int>& v) {
    if (s::ranges::any_of(v, [](int x) { return x <= 0; }))
      throw s::invalid_argument(s::format(
        "[{}] '{}' must be positive but got {}.", mName, what, py_tuple(v)
      ));
  };
  check_positive("kernel_size", this->kernel_size);
  check_positive("stride", this->stride);
  if (s::ranges::any_of(this->padding, [](int p) { return p < 0; }))
    throw s::invalid_argument(s::format(
      "[{}] 'padding' must be non-negative but got {}.", mName,
      py_tuple(this->padding)
    ));
  int k = (int)this->kernel_size.size();
  for (int a = -k - 1; a < -1; ++a) mAxes.push_back(a);
}

s::string Pool::extra_repr() const {
  return s::format(
    "kernel_size={}, stride={}, padding={}",
    py_tuple(kernel_size), py_tuple(stride), py_tuple(padding)
  );
}

mx::array Pool::forward(const mx::array& x) {
  mx::array y = x;
  if (s::ranges::any_of(padding, [](int p) { return p > 0; })) {
    s::vector<s::pair<int, int>> widths{{0, 0}};
    for (int p : padding) widths.emplace_back(p, p);
    widths.emplace_back(0, 0);
    y = mx::pad(y, widths, mx::array(mPaddingValue, y.dtype()));
  }
  return mReduce(sliding_windows(y, kernel_size, stride), mAxes);
}

} // namespace rlbg::nn
