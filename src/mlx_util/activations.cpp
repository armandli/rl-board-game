#include <mlx_util/activations.h>

#include <cmath>

#include <numbers>
#include <stdexcept>

namespace rlbg::nn {

namespace sn = std::numbers;

mx::array sigmoid(const mx::array& x) { return mx::sigmoid(x); }

mx::array relu(const mx::array& x) { return mx::maximum(x, scalar_like(0, x)); }

mx::array relu2(const mx::array& x) { return mx::square(nn::relu(x)); }

mx::array relu6(const mx::array& x) {
  return mx::minimum(nn::relu(x), scalar_like(6.0, x));
}

mx::array leaky_relu(const mx::array& x, double negative_slope) {
  return mx::maximum(scalar_like(negative_slope, x) * x, x);
}

mx::array log_softmax(const mx::array& x, int axis) {
  mx::array y = x - mx::stop_gradient(mx::max(x, axis, true));
  return y - mx::logsumexp(y, axis, true);
}

mx::array elu(const mx::array& x, double alpha) {
  return mx::where(
    mx::greater(x, scalar_like(0, x)),
    x,
    scalar_like(alpha, x) * (mx::exp(x) - scalar_like(1, x))
  );
}

mx::array softmax(const mx::array& x, int axis) { return mx::softmax(x, axis); }

mx::array softplus(const mx::array& x) { return mx::logaddexp(x, scalar_like(0, x)); }

mx::array softsign(const mx::array& x) {
  return mx::divide(x, scalar_like(1, x) + mx::abs(x));
}

mx::array softshrink(const mx::array& x, double lambd) {
  mx::array l = scalar_like(lambd, x);
  return mx::where(
    mx::greater(mx::abs(x), l),
    x - mx::sign(x) * l,
    scalar_like(0, x)
  );
}

mx::array celu(const mx::array& x, double alpha) {
  mx::array a = scalar_like(alpha, x);
  mx::array zero = scalar_like(0, x);
  return mx::maximum(x, zero) +
         a * (mx::exp(mx::minimum(x, zero) / a) - scalar_like(1, x));
}

mx::array silu(const mx::array& x) { return x * mx::sigmoid(x); }

mx::array log_sigmoid(const mx::array& x) { return -nn::softplus(-x); }

mx::array gelu(const mx::array& x) {
  return x * (scalar_like(1, x) + mx::erf(x / scalar_like(sn::sqrt2, x))) /
         scalar_like(2, x);
}

mx::array gelu_approx(const mx::array& x) {
  mx::array inner = scalar_like(std::sqrt(2.0 / sn::pi), x) *
                    (x + scalar_like(0.044715, x) * mx::power(x, scalar_like(3, x)));
  return scalar_like(0.5, x) * x * (scalar_like(1, x) + mx::tanh(inner));
}

mx::array gelu_fast_approx(const mx::array& x) {
  return x * mx::sigmoid(scalar_like(1.702, x) * x);
}

mx::array glu(const mx::array& x, int axis) {
  auto parts = mx::split(x, 2, axis);
  return parts[0] * mx::sigmoid(parts[1]);
}

mx::array step(const mx::array& x, double threshold) {
  return mx::where(
    mx::greater_equal(x, scalar_like(threshold, x)),
    mx::array(1),
    mx::array(0)
  );
}

mx::array selu(const mx::array& x) {
  return nn::elu(x, 1.67326) * scalar_like(1.0507, x);
}

mx::array prelu(const mx::array& x, const mx::array& alpha) {
  mx::array zero = scalar_like(0, x);
  return mx::maximum(zero, x) + alpha * mx::minimum(zero, x);
}

mx::array mish(const mx::array& x) { return x * mx::tanh(nn::softplus(x)); }

mx::array hardswish(const mx::array& x) {
  mx::array max_x_3 = mx::maximum(x + scalar_like(3, x), scalar_like(0, x));
  return x * mx::minimum(max_x_3, scalar_like(6, x)) / scalar_like(6, x);
}

mx::array hard_tanh(const mx::array& x, double min_val, double max_val) {
  return mx::minimum(
    mx::maximum(x, scalar_like(min_val, x)),
    scalar_like(max_val, x)
  );
}

mx::array hard_shrink(const mx::array& x, double lambd) {
  return mx::where(
    mx::greater(mx::abs(x), scalar_like(lambd, x)),
    x,
    scalar_like(0, x)
  );
}

mx::array softmin(const mx::array& x, int axis) { return mx::softmax(-x, axis); }

mx::array tanh(const mx::array& x) { return mx::tanh(x); }

GELU::GELU(s::string approx) : approx(s::move(approx)) {
  if (this->approx != "none" and this->approx != "precise" and
      this->approx != "tanh" and this->approx != "fast")
    throw s::invalid_argument(
      "The approximation should be in ['none', 'precise', 'tanh', 'fast'] "
      "but '" + this->approx + "' was given"
    );
}

mx::array GELU::forward(const mx::array& x) {
  if (approx == "none") return nn::gelu(x);
  if (approx == "precise" or approx == "tanh") return nn::gelu_approx(x);
  return nn::gelu_fast_approx(x);
}

} // namespace rlbg::nn
