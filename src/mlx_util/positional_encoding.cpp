#include <mlx_util/positional_encoding.h>

#include <cmath>

#include <format>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <mlx_util/repr.h>

namespace rlbg::nn {

namespace {

mx::array make_sigmas(int dims, double min_freq, double max_freq, bool full_turns) {
  if (dims <= 0 or dims % 2 != 0)
    throw s::invalid_argument(s::format(
      "[SinusoidalPositionalEncoding] dims must be positive and even but got {}.",
      dims
    ));
  int half = dims / 2;
  mx::array one_zero = dims == 2
    ? mx::array({1.0f})
    : mx::array(1.0f) - mx::arange(0, half) / mx::array((float)(half - 1));
  double lmin = std::log(min_freq);
  double lmax = std::log(max_freq);
  mx::array sigmas = mx::exp(
    one_zero * mx::array((float)(lmax - lmin)) + mx::array((float)lmin)
  );
  if (full_turns) sigmas = sigmas * mx::array((float)(2 * s::numbers::pi));
  return sigmas;
}

// slopes of the ALiBi paper: a geometric sequence per power of two of heads
s::vector<double> alibi_slopes(int n) {
  double lg = std::log2((double)n);
  if (lg == std::floor(lg)) {
    double start = std::pow(2.0, -std::pow(2.0, -(lg - 3)));
    s::vector<double> out;
    for (int i = 0; i < n; ++i) out.push_back(start * std::pow(start, i));
    return out;
  }
  int closest = (int)std::pow(2.0, std::floor(lg));
  s::vector<double> out = alibi_slopes(closest);
  s::vector<double> extra = alibi_slopes(2 * closest);
  for (int i = 0; i < n - closest; ++i) out.push_back(extra[2 * i]);
  return out;
}

} // namespace

s::string RoPE::extra_repr() const {
  return s::format("{}, traditional={}", dims, py_bool(traditional));
}

SinusoidalPositionalEncoding::SinusoidalPositionalEncoding(
  int dims,
  double min_freq,
  double max_freq,
  s::optional<double> scale,
  bool cos_first,
  bool full_turns)
  : scale(scale ? *scale : std::sqrt(2.0 / dims)),
    cos_first(cos_first),
    mSigmas(make_sigmas(dims, min_freq, max_freq, full_turns))
{}

mx::array SinusoidalPositionalEncoding::forward(const mx::array& x) {
  mx::array y = mx::expand_dims(x, -1) * mSigmas;
  mx::array cosy = mx::cos(y);
  mx::array siny = mx::sin(y);
  y = cos_first ? mx::concatenate({cosy, siny}, -1)
                : mx::concatenate({siny, cosy}, -1);
  if (scale != 1.0) y = y * scalar_like(scale, y);
  return y;
}

mx::array ALiBi::create_alibi_matrix(
  int q_sequence_length,
  int k_sequence_length,
  int num_heads,
  int offset,
  mx::Dtype dtype)
{
  mx::array x1 = mx::arange(offset, q_sequence_length);
  mx::array x2 = mx::arange(0, k_sequence_length);
  mx::array distance = -mx::abs(mx::expand_dims(
    mx::expand_dims(x1, 1) - mx::expand_dims(x2, 0),
    {0, 1}
  ));
  mx::array slope = create_alibi_slope(num_heads, dtype);
  return mx::astype(distance * slope, dtype);
}

mx::array ALiBi::create_alibi_slope(int num_heads, mx::Dtype dtype) {
  s::vector<double> slopes = alibi_slopes(num_heads);
  s::vector<float> values(slopes.begin(), slopes.end());
  mx::array out = mx::astype(
    mx::array(values.data(), {(int)values.size()}, mx::float32),
    dtype
  );
  return mx::expand_dims(out, {-1, -2});
}

mx::array ALiBi::forward(
  const mx::array& attention_scores,
  int offset,
  const s::optional<mx::array>& mask) const
{
  mx::array alibi_mask = create_alibi_matrix(
    attention_scores.shape(-2) + offset,
    attention_scores.shape(-1),
    attention_scores.shape(1),
    offset,
    attention_scores.dtype()
  );
  if (mask) alibi_mask = alibi_mask + *mask;
  return attention_scores + alibi_mask;
}

} // namespace rlbg::nn
