#include <mlx_util/linear.h>

#include <cmath>

#include <format>

#include <mlx_util/repr.h>

namespace rlbg::nn {

Linear::Linear(int input_dims, int output_dims, bool bias)
  : weight(mx::random::uniform(
      -std::sqrt(1.0f / input_dims),
      std::sqrt(1.0f / input_dims),
      {output_dims, input_dims}
    ))
{
  register_parameter("weight", weight);
  if (bias) {
    float scale = std::sqrt(1.0f / input_dims);
    this->bias = mx::random::uniform(-scale, scale, {output_dims});
    register_parameter("bias", *this->bias);
  }
}

s::string Linear::extra_repr() const {
  return s::format(
    "input_dims={}, output_dims={}, bias={}",
    weight.shape(1), weight.shape(0), py_bool(bias.has_value())
  );
}

mx::array Linear::forward(const mx::array& x) {
  if (bias) return mx::addmm(*bias, x, mx::transpose(weight));
  return mx::matmul(x, mx::transpose(weight));
}

Bilinear::Bilinear(int input1_dims, int input2_dims, int output_dims, bool bias)
  : weight(mx::random::uniform(
      -std::sqrt(1.0f / input1_dims),
      std::sqrt(1.0f / input1_dims),
      {output_dims, input2_dims, input1_dims}
    ))
{
  register_parameter("weight", weight);
  if (bias) {
    float scale = std::sqrt(1.0f / input1_dims);
    this->bias = mx::random::uniform(-scale, scale, {output_dims});
    register_parameter("bias", *this->bias);
  }
}

s::string Bilinear::extra_repr() const {
  return s::format(
    "input1_dims={}, input2_dims={}, output_dims={}, bias={}",
    weight.shape(2), weight.shape(1), weight.shape(0),
    py_bool(bias.has_value())
  );
}

mx::array Bilinear::forward(const mx::array& x1, const mx::array& x2) {
  int out = weight.shape(0);
  int in2 = weight.shape(1);
  int in1 = weight.shape(2);
  mx::Shape xshape(x1.shape().begin(), x1.shape().end() - 1);
  mx::array a = mx::reshape(x1, {-1, in1});
  mx::array b = mx::reshape(x2, {-1, 1, in2});
  mx::array w = mx::reshape(weight, {out * in2, in1});
  mx::array y = mx::matmul(a, mx::transpose(w));
  y = mx::swapaxes(mx::reshape(y, {-1, out, in2}), -2, -1);
  y = mx::squeeze(mx::matmul(b, y), 1);
  xshape.push_back(out);
  y = mx::reshape(y, xshape);
  if (bias) y = y + *bias;
  return y;
}

} // namespace rlbg::nn
