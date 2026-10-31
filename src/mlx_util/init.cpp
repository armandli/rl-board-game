#include <mlx_util/init.h>

#include <cmath>

#include <stdexcept>
#include <utility>

namespace rlbg::nn::init {

namespace {

s::pair<double, double> fan_in_fan_out(const mx::array& x) {
  if (x.ndim() < 2)
    throw s::invalid_argument(
      "Glorot / He initialization requires at least 2 dimensional input but "
      "input with " + s::to_string(x.ndim()) + " dimensions."
    );
  double fan_in = x.shape(-1);
  double fan_out = x.shape(0);
  if (x.ndim() > 2) {
    double receptive_field = 1;
    for (size_t d = 1; d + 1 < x.ndim(); ++d) receptive_field *= x.shape(d);
    fan_in *= receptive_field;
    fan_out *= receptive_field;
  }
  return {fan_in, fan_out};
}

double select_fan(const mx::array& a, const s::string& mode) {
  auto [fan_in, fan_out] = fan_in_fan_out(a);
  if (mode == "fan_in") return fan_in;
  if (mode == "fan_out") return fan_out;
  throw s::invalid_argument("Invalid mode: " + mode + ". Valid modes are: fan_in, fan_out");
}

mx::array normal_like(const mx::array& a, double mean, double std, mx::Dtype dtype) {
  return mx::random::normal(a.shape(), dtype, (float)mean, (float)std);
}

mx::array uniform_like(const mx::array& a, double low, double high, mx::Dtype dtype) {
  return mx::random::uniform(
    mx::array((float)low), mx::array((float)high), a.shape(), dtype
  );
}

} // namespace

Initializer constant(double value, mx::Dtype dtype) {
  return [=](const mx::array& a) { return mx::full(a.shape(), mx::array(value), dtype); };
}

Initializer normal(double mean, double std, mx::Dtype dtype) {
  return [=](const mx::array& a) { return normal_like(a, mean, std, dtype); };
}

Initializer uniform(double low, double high, mx::Dtype dtype) {
  return [=](const mx::array& a) { return uniform_like(a, low, high, dtype); };
}

Initializer identity(mx::Dtype dtype) {
  return [=](const mx::array& a) {
    if (a.ndim() != 2 or a.shape(0) != a.shape(1))
      throw s::invalid_argument("The input array must be a square matrix.");
    return mx::eye(a.shape(0), dtype);
  };
}

Initializer glorot_normal(mx::Dtype dtype, double gain) {
  return [=](const mx::array& a) {
    auto [fan_in, fan_out] = fan_in_fan_out(a);
    return normal_like(a, 0.0, gain * std::sqrt(2.0 / (fan_in + fan_out)), dtype);
  };
}

Initializer glorot_uniform(mx::Dtype dtype, double gain) {
  return [=](const mx::array& a) {
    auto [fan_in, fan_out] = fan_in_fan_out(a);
    double limit = gain * std::sqrt(6.0 / (fan_in + fan_out));
    return uniform_like(a, -limit, limit, dtype);
  };
}

Initializer he_normal(mx::Dtype dtype, const s::string& mode, double gain) {
  return [=](const mx::array& a) {
    return normal_like(a, 0.0, gain / std::sqrt(select_fan(a, mode)), dtype);
  };
}

Initializer he_uniform(mx::Dtype dtype, const s::string& mode, double gain) {
  return [=](const mx::array& a) {
    double limit = gain * std::sqrt(3.0 / select_fan(a, mode));
    return uniform_like(a, -limit, limit, dtype);
  };
}

Initializer sparse(double sparsity, double mean, double std, mx::Dtype dtype) {
  return [=](const mx::array& a) {
    if (a.ndim() != 2)
      throw s::invalid_argument("Only tensors with 2 dimensions are supported");
    int cols = a.shape(1);
    int num_zeros = (int)std::ceil(sparsity * cols);
    mx::array order = mx::argsort(mx::random::uniform(a.shape()), 1);
    mx::array out = normal_like(a, mean, std, dtype);
    if (num_zeros == 0) return out;
    mx::array zero_at = mx::slice(order, {0, 0}, {a.shape(0), num_zeros});
    return mx::put_along_axis(out, zero_at, mx::zeros({1}, dtype), 1);
  };
}

Initializer orthogonal(double gain, mx::Dtype dtype) {
  return [=](const mx::array& a) {
    if (a.ndim() != 2)
      throw s::invalid_argument(
        "Orthogonal initialization requires a 2D array but got a " +
        s::to_string(a.ndim()) + "D array."
      );
    int rows = a.shape(0);
    int cols = a.shape(1);
    int n = s::max(rows, cols);
    mx::array rmat = mx::random::normal({n, n});
    auto [q, r] = mx::linalg::qr(rmat, mx::Device(mx::Device::cpu));
    q = q * mx::sign(mx::diag(r));
    q = mx::slice(q, {0, 0}, {rows, cols}) * mx::array((float)gain);
    return mx::astype(q, dtype);
  };
}

} // namespace rlbg::nn::init
