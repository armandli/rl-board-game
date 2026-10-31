#include <mlx_util/convolution.h>

#include <cmath>

#include <format>
#include <stdexcept>

namespace rlbg::nn {

namespace {

// fan_in is the number of inputs one output element sees
mx::array uniform_weight(int fan_in, const mx::Shape& shape) {
  float scale = std::sqrt(1.0f / fan_in);
  return mx::random::uniform(-scale, scale, shape);
}

s::optional<mx::array> zero_bias(bool bias, int out_channels) {
  if (not bias) return s::nullopt;
  return mx::zeros({out_channels});
}

void check_groups(int in_channels, int groups) {
  if (in_channels % groups != 0)
    throw s::invalid_argument(s::format(
      "The number of input channels ({}) must be divisible by the number of "
      "groups ({})", in_channels, groups
    ));
}

mx::array add_bias(const mx::array& y, const s::optional<mx::array>& bias) {
  return bias ? y + *bias : y;
}

s::string shape_slice(const mx::array& w, int begin, int end) {
  s::vector<int> dims;
  for (int i = begin; i < end; ++i) dims.push_back(w.shape(i));
  return py_tuple(dims);
}

} // namespace

Conv1d::Conv1d(
  int in_channels,
  int out_channels,
  int kernel_size,
  int stride,
  int padding,
  int dilation,
  int groups,
  bool bias)
  : weight(uniform_weight(
      (check_groups(in_channels, groups), in_channels / groups * kernel_size),
      {out_channels, kernel_size, in_channels / groups}
    )),
    bias(zero_bias(bias, out_channels)),
    padding(padding),
    dilation(dilation),
    stride(stride),
    groups(groups)
{
  register_parameter("weight", weight);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string Conv1d::extra_repr() const {
  return s::format(
    "{}, {}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, "
    "bias={}",
    weight.shape(-1) * groups, weight.shape(0), weight.shape(1), stride,
    padding, dilation, groups, py_bool(bias.has_value())
  );
}

mx::array Conv1d::forward(const mx::array& x) {
  return add_bias(mx::conv1d(x, weight, stride, padding, dilation, groups), bias);
}

Conv2d::Conv2d(
  int in_channels,
  int out_channels,
  Int2 kernel_size,
  Int2 stride,
  Int2 padding,
  Int2 dilation,
  int groups,
  bool bias)
  : weight(uniform_weight(
      (check_groups(in_channels, groups),
       in_channels / groups * kernel_size[0] * kernel_size[1]),
      {out_channels, kernel_size[0], kernel_size[1], in_channels / groups}
    )),
    bias(zero_bias(bias, out_channels)),
    padding(padding),
    stride(stride),
    dilation(dilation),
    groups(groups)
{
  register_parameter("weight", weight);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string Conv2d::extra_repr() const {
  return s::format(
    "{}, {}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, "
    "bias={}",
    weight.shape(-1) * groups, weight.shape(0), shape_slice(weight, 1, 3),
    stride.repr(), padding.repr(), dilation.repr(), groups,
    py_bool(bias.has_value())
  );
}

mx::array Conv2d::forward(const mx::array& x) {
  return add_bias(
    mx::conv2d(x, weight, stride.pair(), padding.pair(), dilation.pair(), groups),
    bias
  );
}

Conv3d::Conv3d(
  int in_channels,
  int out_channels,
  Int3 kernel_size,
  Int3 stride,
  Int3 padding,
  Int3 dilation,
  bool bias)
  : weight(uniform_weight(
      in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
      {out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels}
    )),
    bias(zero_bias(bias, out_channels)),
    padding(padding),
    stride(stride),
    dilation(dilation)
{
  register_parameter("weight", weight);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string Conv3d::extra_repr() const {
  return s::format(
    "{}, {}, kernel_size={}, stride={}, padding={}, dilation={}, bias={}",
    weight.shape(-1), weight.shape(0), shape_slice(weight, 1, 4),
    stride.repr(), padding.repr(), dilation.repr(), py_bool(bias.has_value())
  );
}

mx::array Conv3d::forward(const mx::array& x) {
  return add_bias(
    mx::conv3d(x, weight, stride.tuple(), padding.tuple(), dilation.tuple()),
    bias
  );
}

ConvTranspose1d::ConvTranspose1d(
  int in_channels,
  int out_channels,
  int kernel_size,
  int stride,
  int padding,
  int dilation,
  int output_padding,
  bool bias)
  : weight(uniform_weight(
      in_channels * kernel_size,
      {out_channels, kernel_size, in_channels}
    )),
    bias(zero_bias(bias, out_channels)),
    padding(padding),
    dilation(dilation),
    stride(stride),
    output_padding(output_padding)
{
  register_parameter("weight", weight);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string ConvTranspose1d::extra_repr() const {
  return s::format(
    "{}, {}, kernel_size={}, stride={}, padding={}, dilation={}, "
    "output_padding={}, bias={}",
    weight.shape(-1), weight.shape(0), weight.shape(1), stride, padding,
    dilation, output_padding, py_bool(bias.has_value())
  );
}

mx::array ConvTranspose1d::forward(const mx::array& x) {
  return add_bias(
    mx::conv_transpose1d(x, weight, stride, padding, dilation, output_padding),
    bias
  );
}

ConvTranspose2d::ConvTranspose2d(
  int in_channels,
  int out_channels,
  Int2 kernel_size,
  Int2 stride,
  Int2 padding,
  Int2 dilation,
  Int2 output_padding,
  bool bias)
  : weight(uniform_weight(
      in_channels * kernel_size[0] * kernel_size[1],
      {out_channels, kernel_size[0], kernel_size[1], in_channels}
    )),
    bias(zero_bias(bias, out_channels)),
    padding(padding),
    stride(stride),
    dilation(dilation),
    output_padding(output_padding)
{
  register_parameter("weight", weight);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string ConvTranspose2d::extra_repr() const {
  return s::format(
    "{}, {}, kernel_size={}, stride={}, padding={}, dilation={}, "
    "output_padding={}, bias={}",
    weight.shape(-1), weight.shape(0), shape_slice(weight, 1, 3),
    stride.repr(), padding.repr(), dilation.repr(), output_padding.repr(),
    py_bool(bias.has_value())
  );
}

mx::array ConvTranspose2d::forward(const mx::array& x) {
  return add_bias(
    mx::conv_transpose2d(
      x, weight, stride.pair(), padding.pair(), dilation.pair(),
      output_padding.pair()
    ),
    bias
  );
}

ConvTranspose3d::ConvTranspose3d(
  int in_channels,
  int out_channels,
  Int3 kernel_size,
  Int3 stride,
  Int3 padding,
  Int3 dilation,
  Int3 output_padding,
  bool bias)
  : weight(uniform_weight(
      in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
      {out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels}
    )),
    bias(zero_bias(bias, out_channels)),
    padding(padding),
    stride(stride),
    dilation(dilation),
    output_padding(output_padding)
{
  register_parameter("weight", weight);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string ConvTranspose3d::extra_repr() const {
  return s::format(
    "{}, {}, kernel_size={}, stride={}, padding={}, dilation={}, "
    "output_padding={}, bias={}",
    weight.shape(-1), weight.shape(0), shape_slice(weight, 1, 4),
    stride.repr(), padding.repr(), dilation.repr(), output_padding.repr(),
    py_bool(bias.has_value())
  );
}

mx::array ConvTranspose3d::forward(const mx::array& x) {
  return add_bias(
    mx::conv_transpose3d(
      x, weight, stride.tuple(), padding.tuple(), dilation.tuple(),
      output_padding.tuple()
    ),
    bias
  );
}

} // namespace rlbg::nn
