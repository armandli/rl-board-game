#include <mlx_util/upsample.h>

#include <cmath>

#include <algorithm>
#include <format>
#include <functional>
#include <stdexcept>
#include <utility>

#include <mlx_util/repr.h>

namespace rlbg::nn {

namespace {

// sample indices along one axis and their weights, shaped to broadcast
// against (N, *spatial, C)
using IndexWeight = s::pair<mx::array, mx::array>;
using IndicesFn = s::function<
  s::vector<IndexWeight>(int n, double scale, bool align_corners, int dim, int ndims)
>;

mx::array axis_shaped(const mx::array& v, int dim, int ndims) {
  mx::Shape shape(ndims, 1);
  shape[dim] = -1;
  return mx::reshape(v, shape);
}

mx::array f32(double v) { return mx::array((float)v); }

mx::array scaled_indices(int n, double scale, bool align_corners, int dim, int ndims) {
  int m = (int)(scale * n);
  mx::array idx = mx::arange(m, mx::float32);
  if (align_corners) {
    idx = idx * f32((double)(n - 1) / s::max(m - 1, 1));
  } else {
    double step = 1.0 / scale;
    double start = ((m - 1) * step - n + 1) / 2;
    idx = idx * f32(step) - f32(start);
  }
  return axis_shaped(idx, dim, ndims);
}

mx::array nearest_indices(int n, double scale, int dim, int ndims) {
  int m = (int)(scale * n);
  mx::array idx = mx::arange(m, mx::float32);
  if (m > n)
    idx = mx::round((idx + f32(0.5)) * f32((double)n / m) - f32(0.5));
  else
    idx = idx * f32((double)n / m);
  return axis_shaped(mx::astype(idx, mx::uint32), dim, ndims);
}

s::vector<IndexWeight> linear_indices(int n, double scale, bool align_corners, int dim, int ndims) {
  mx::array idx = mx::clip(
    scaled_indices(n, scale, align_corners, dim, ndims),
    f32(0),
    f32(n - 1)
  );
  mx::array left = mx::floor(idx);
  mx::array right = mx::ceil(idx);
  mx::array weight = mx::expand_dims(idx - left, -1);
  return {
    {mx::astype(left, mx::uint32), f32(1) - weight},
    {mx::astype(right, mx::uint32), weight},
  };
}

mx::array triangle_kernel(const mx::array& x) {
  return mx::maximum(f32(1) - x, f32(0));
}

mx::array cubic_kernel(const mx::array& x) {
  const double a = -0.5;
  mx::array inner = (f32(a + 2.0) * x - f32(a + 3.0)) * x * x + f32(1);
  mx::array outer = (((x - f32(5)) * x + f32(8)) * x - f32(4)) * f32(a);
  return mx::where(
    mx::less_equal(x, f32(1)),
    inner,
    mx::where(mx::less_equal(x, f32(2)), outer, f32(0))
  );
}

s::vector<IndexWeight> aa_indices(
  int n,
  double scale,
  bool align_corners,
  int dim,
  int ndims,
  const ArrayFn& kernel,
  double kernel_radius)
{
  mx::array indices = scaled_indices(n, scale, align_corners, dim, ndims);
  double inv_scale = scale < 1 ? 1.0 / scale : 1.0;
  int num_taps = (int)std::ceil(kernel_radius * inv_scale) + 1;

  s::vector<mx::array> all_idx;
  s::vector<mx::array> all_w;
  for (int k = -num_taps + 1; k < num_taps; ++k) {
    mx::array idx = mx::floor(indices) + f32(k);
    mx::array dist = mx::abs(indices - idx) / f32(inv_scale);
    mx::array inside = mx::logical_and(
      mx::greater_equal(idx, f32(0)),
      mx::less(idx, f32(n))
    );
    all_idx.push_back(idx);
    all_w.push_back(mx::where(inside, kernel(dist), f32(0)));
  }
  mx::array w_sum = all_w[0];
  for (size_t i = 1; i < all_w.size(); ++i) w_sum = w_sum + all_w[i];
  w_sum = mx::where(mx::greater(w_sum, f32(0)), w_sum, f32(1));

  s::vector<IndexWeight> out;
  for (size_t i = 0; i < all_idx.size(); ++i)
    out.emplace_back(
      mx::astype(mx::clip(all_idx[i], f32(0), f32(n - 1)), mx::uint32),
      mx::expand_dims(all_w[i] / w_sum, -1)
    );
  return out;
}

s::vector<IndexWeight> linear_aa_indices(int n, double scale, bool align_corners, int dim, int ndims) {
  return aa_indices(n, scale, align_corners, dim, ndims, triangle_kernel, 1.0);
}

s::vector<IndexWeight> cubic_aa_indices(int n, double scale, bool align_corners, int dim, int ndims) {
  return aa_indices(n, scale, align_corners, dim, ndims, cubic_kernel, 2.0);
}

s::vector<IndexWeight> cubic_indices(int n, double scale, bool align_corners, int dim, int ndims) {
  mx::array indices = scaled_indices(n, scale, align_corners, dim, ndims);
  mx::array l1 = mx::floor(indices);
  mx::array r1 = mx::floor(indices + f32(1));
  mx::array l2 = l1 - f32(1);
  mx::array r2 = r1 + f32(1);

  // Keys cubic convolution with a = -0.75; dist selects the inner (|x| <= 1)
  // or outer (1 < |x| <= 2) piece
  auto weight = [&](const mx::array& grid, int dist) {
    const double a = -0.75;
    mx::array x = mx::abs(indices - grid);
    mx::array w = dist == 1
      ? (f32(a + 2.0) * x - f32(a + 3.0)) * x * x + f32(1)
      : (((x - f32(5)) * x + f32(8)) * x - f32(4)) * f32(a);
    return mx::expand_dims(w, -1);
  };
  auto clipped = [&](const mx::array& idx) {
    return mx::astype(mx::clip(idx, f32(0), f32(n - 1)), mx::uint32);
  };
  return {
    {clipped(l1), weight(l1, 1)},
    {clipped(r1), weight(r1, 1)},
    {clipped(l2), weight(l2, 2)},
    {clipped(r2), weight(r2, 2)},
  };
}

void check_dims(const mx::array& x, const s::vector<double>& scale_factor) {
  if (x.ndim() - 2 != scale_factor.size())
    throw s::invalid_argument(s::format(
      "A scale needs to be provided for each spatial dimension but got {} "
      "scales for a {}D input", scale_factor.size(), x.ndim() - 2
    ));
}

void validate_antialias_align_corners(bool align_corners, bool antialias) {
  if (antialias and align_corners)
    throw s::invalid_argument(
      "[Upsample] antialias=True with align_corners=True is not supported. "
      "Use align_corners=False for antialiased interpolation."
    );
}

mx::array interpolate(
  const mx::array& x,
  const s::vector<double>& scale_factor,
  const IndicesFn& indices_fn,
  bool align_corners)
{
  check_dims(x, scale_factor);
  int dims = (int)x.ndim() - 2;
  s::vector<s::vector<IndexWeight>> per_axis;
  for (int i = 0; i < dims; ++i)
    per_axis.push_back(indices_fn(x.shape(i + 1), scale_factor[i], align_corners, i, dims));

  // sum over the cartesian product of every axis' taps
  s::optional<mx::array> total;
  s::vector<size_t> pick(dims, 0);
  while (true) {
    mx::array sample = x;
    s::optional<mx::array> weight;
    for (int i = 0; i < dims; ++i) {
      const IndexWeight& iw = per_axis[i][pick[i]];
      sample = mx::take(sample, mx::flatten(iw.first), i + 1);
      weight = weight ? *weight * iw.second : iw.second;
    }
    mx::array term = *weight * sample;
    total = total ? *total + term : term;

    int axis = dims - 1;
    while (axis >= 0 and ++pick[axis] == per_axis[axis].size()) pick[axis--] = 0;
    if (axis < 0) break;
  }
  return *total;
}

mx::array interpolate_separable(
  const mx::array& x,
  const s::vector<double>& scale_factor,
  const IndicesFn& indices_fn,
  bool align_corners)
{
  check_dims(x, scale_factor);
  int dims = (int)x.ndim() - 2;
  mx::array out = x;
  for (int i = 0; i < dims; ++i) {
    s::optional<mx::array> sum;
    for (const auto& [idx, w] : indices_fn(x.shape(i + 1), scale_factor[i], align_corners, i, dims)) {
      mx::array sample = mx::take(out, mx::flatten(idx), i + 1) * w;
      sum = sum ? *sum + sample : sample;
    }
    out = *sum;
  }
  return out;
}

} // namespace

mx::array upsample_nearest(const mx::array& x, const s::vector<double>& scale_factor) {
  check_dims(x, scale_factor);
  int dims = (int)x.ndim() - 2;
  bool integral = s::ranges::all_of(scale_factor, [](double v) {
    return v == std::floor(v);
  });
  if (integral) {
    s::vector<int> shape(x.shape().begin(), x.shape().end());
    for (int d = 0; d < dims; ++d) shape.insert(shape.begin() + 2 + 2 * d, 1);
    mx::array y = mx::reshape(x, mx::Shape(shape.begin(), shape.end()));
    for (int d = 0; d < dims; ++d) shape[2 + 2 * d] = (int)scale_factor[d];
    y = mx::broadcast_to(y, mx::Shape(shape.begin(), shape.end()));
    for (int d = 0; d < dims; ++d) {
      shape[d + 1] *= shape[d + 2];
      shape.erase(shape.begin() + d + 2);
    }
    return mx::reshape(y, mx::Shape(shape.begin(), shape.end()));
  }
  mx::array y = x;
  for (int i = 0; i < dims; ++i)
    y = mx::take(
      y,
      mx::flatten(nearest_indices(x.shape(i + 1), scale_factor[i], i, dims)),
      i + 1
    );
  return y;
}

mx::array upsample_linear(
  const mx::array& x,
  const s::vector<double>& scale_factor,
  bool align_corners,
  bool antialias)
{
  validate_antialias_align_corners(align_corners, antialias);
  if (antialias)
    return interpolate_separable(x, scale_factor, linear_aa_indices, align_corners);
  if (not align_corners)
    return interpolate_separable(x, scale_factor, linear_indices, align_corners);
  return interpolate(x, scale_factor, linear_indices, align_corners);
}

mx::array upsample_cubic(
  const mx::array& x,
  const s::vector<double>& scale_factor,
  bool align_corners,
  bool antialias)
{
  validate_antialias_align_corners(align_corners, antialias);
  if (antialias)
    return interpolate_separable(x, scale_factor, cubic_aa_indices, align_corners);
  return interpolate(x, scale_factor, cubic_indices, align_corners);
}

Upsample::Upsample(
  s::vector<double> scale_factor,
  s::string mode,
  bool align_corners,
  bool antialias)
  : scale_factor(s::move(scale_factor)),
    mode(s::move(mode)),
    align_corners(align_corners),
    antialias(antialias)
{
  if (this->mode != "nearest" and this->mode != "linear" and this->mode != "cubic")
    throw s::invalid_argument("[Upsample] Got unsupported upsampling algorithm: " + this->mode);
  if (antialias and this->mode == "nearest")
    throw s::invalid_argument(
      "[Upsample] Antialiasing is not supported for nearest neighbor upsampling"
    );
  validate_antialias_align_corners(align_corners, antialias);
}

s::string Upsample::extra_repr() const {
  s::string scale;
  if (mScalar) {
    scale = py_float(scale_factor[0]);
  } else {
    s::vector<s::string> parts;
    for (double v : scale_factor) parts.push_back(py_float(v));
    scale = "(";
    for (size_t i = 0; i < parts.size(); ++i) scale += (i ? ", " : "") + parts[i];
    scale += parts.size() == 1 ? ",)" : ")";
  }
  s::string out = s::format(
    "scale_factor={}, mode='{}', align_corners={}",
    scale, mode, py_bool(align_corners)
  );
  if (antialias) out += ", antialias=True";
  return out;
}

mx::array Upsample::forward(const mx::array& x) {
  int dims = (int)x.ndim() - 2;
  if (dims <= 0)
    throw s::invalid_argument(s::format(
      "[Upsample] The input should have at least 1 spatial dimension which "
      "means it should be at least 3D but {}D was provided", x.ndim()
    ));
  s::vector<double> scales = scale_factor;
  if (mScalar) {
    scales.assign(dims, scale_factor[0]);
  } else if ((int)scales.size() != dims) {
    throw s::invalid_argument(s::format(
      "[Upsample] One scale per spatial dimension is required but scale_factor "
      "has {} entries for a {}D input", scales.size(), x.ndim()
    ));
  }
  if (mode == "nearest") return upsample_nearest(x, scales);
  if (mode == "linear") return upsample_linear(x, scales, align_corners, antialias);
  return upsample_cubic(x, scales, align_corners, antialias);
}

} // namespace rlbg::nn
