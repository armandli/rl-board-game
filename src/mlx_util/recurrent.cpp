#include <mlx_util/recurrent.h>

#include <cmath>

#include <format>
#include <vector>

#include <mlx_util/activations.h>
#include <mlx_util/repr.h>

namespace rlbg::nn {

namespace {

mx::array uniform(int hidden_size, const mx::Shape& shape) {
  float scale = 1.0f / std::sqrt((float)hidden_size);
  return mx::random::uniform(-scale, scale, shape);
}

s::optional<mx::array> maybe_uniform(bool use, int hidden_size, const mx::Shape& shape) {
  if (not use) return s::nullopt;
  return uniform(hidden_size, shape);
}

mx::array project(
  const mx::array& x,
  const mx::array& w,
  const s::optional<mx::array>& bias)
{
  return bias ? mx::addmm(*bias, x, mx::transpose(w)) : mx::matmul(x, mx::transpose(w));
}

// x[..., idx, :]
mx::array time_step(const mx::array& x, int idx) { return mx::take(x, idx, -2); }

} // namespace

RNN::RNN(int input_size, int hidden_size, bool bias, ArrayFn nonlinearity)
  : Wxh(uniform(hidden_size, {hidden_size, input_size})),
    Whh(uniform(hidden_size, {hidden_size, hidden_size})),
    bias(maybe_uniform(bias, hidden_size, {hidden_size})),
    nonlinearity(nonlinearity ? s::move(nonlinearity) : ArrayFn(nn::tanh)),
    hidden_size(hidden_size)
{
  register_parameter("Wxh", Wxh);
  register_parameter("Whh", Whh);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string RNN::extra_repr() const {
  return s::format(
    "input_dims={}, hidden_size={}, bias={}",
    Wxh.shape(1), hidden_size, py_bool(bias.has_value())
  );
}

mx::array RNN::forward(const mx::array& x, s::optional<mx::array> hidden) {
  mx::array xp = project(x, Wxh, bias);
  s::vector<mx::array> all_hidden;
  for (int idx = 0; idx < xp.shape(-2); ++idx) {
    mx::array h = hidden
      ? mx::addmm(time_step(xp, idx), *hidden, mx::transpose(Whh))
      : time_step(xp, idx);
    hidden = nonlinearity(h);
    all_hidden.push_back(*hidden);
  }
  return mx::stack(all_hidden, -2);
}

GRU::GRU(int input_size, int hidden_size, bool bias)
  : Wx(uniform(hidden_size, {3 * hidden_size, input_size})),
    Wh(uniform(hidden_size, {3 * hidden_size, hidden_size})),
    b(maybe_uniform(bias, hidden_size, {3 * hidden_size})),
    bhn(maybe_uniform(bias, hidden_size, {hidden_size})),
    hidden_size(hidden_size)
{
  register_parameter("Wx", Wx);
  register_parameter("Wh", Wh);
  if (b) register_parameter("b", *b);
  if (bhn) register_parameter("bhn", *bhn);
}

s::string GRU::extra_repr() const {
  return s::format(
    "input_dims={}, hidden_size={}, bias={}",
    Wx.shape(1), hidden_size, py_bool(b.has_value())
  );
}

mx::array GRU::forward(const mx::array& x, s::optional<mx::array> hidden) {
  mx::array xp = project(x, Wx, b);
  int split = xp.shape(-1) - hidden_size;
  mx::array x_rz = mx::split(xp, mx::Shape{split}, -1)[0];
  mx::array x_n = mx::split(xp, mx::Shape{split}, -1)[1];

  s::vector<mx::array> all_hidden;
  for (int idx = 0; idx < xp.shape(-2); ++idx) {
    mx::array rz = time_step(x_rz, idx);
    s::optional<mx::array> h_proj_n;
    if (hidden) {
      mx::array h_proj = mx::matmul(*hidden, mx::transpose(Wh));
      auto parts = mx::split(h_proj, mx::Shape{h_proj.shape(-1) - hidden_size}, -1);
      h_proj_n = bhn ? parts[1] + *bhn : parts[1];
      rz = rz + parts[0];
    }
    rz = mx::sigmoid(rz);
    auto rz_parts = mx::split(rz, 2, -1);
    mx::array& r = rz_parts[0];
    mx::array& z = rz_parts[1];

    mx::array n = time_step(x_n, idx);
    if (hidden)   n = n + r * *h_proj_n;
    else if (bhn) n = n + r * *bhn;
    n = mx::tanh(n);

    mx::array one = scalar_like(1, z);
    hidden = hidden ? (one - z) * n + z * *hidden : (one - z) * n;
    all_hidden.push_back(*hidden);
  }
  return mx::stack(all_hidden, -2);
}

LSTM::LSTM(int input_size, int hidden_size, bool bias)
  : Wx(uniform(hidden_size, {4 * hidden_size, input_size})),
    Wh(uniform(hidden_size, {4 * hidden_size, hidden_size})),
    bias(maybe_uniform(bias, hidden_size, {4 * hidden_size})),
    hidden_size(hidden_size)
{
  register_parameter("Wx", Wx);
  register_parameter("Wh", Wh);
  if (this->bias) register_parameter("bias", *this->bias);
}

s::string LSTM::extra_repr() const {
  return s::format(
    "input_dims={}, hidden_size={}, bias={}",
    Wx.shape(1), hidden_size, py_bool(bias.has_value())
  );
}

s::pair<mx::array, mx::array> LSTM::forward(
  const mx::array& x,
  s::optional<mx::array> hidden,
  s::optional<mx::array> cell)
{
  mx::array xp = project(x, Wx, bias);
  s::vector<mx::array> all_hidden;
  s::vector<mx::array> all_cell;
  for (int idx = 0; idx < xp.shape(-2); ++idx) {
    mx::array ifgo = time_step(xp, idx);
    if (hidden) ifgo = mx::addmm(ifgo, *hidden, mx::transpose(Wh));
    auto gates = mx::split(ifgo, 4, -1);
    mx::array i = mx::sigmoid(gates[0]);
    mx::array f = mx::sigmoid(gates[1]);
    mx::array g = mx::tanh(gates[2]);
    mx::array o = mx::sigmoid(gates[3]);
    cell = cell ? f * *cell + i * g : i * g;
    hidden = o * mx::tanh(*cell);
    all_cell.push_back(*cell);
    all_hidden.push_back(*hidden);
  }
  return {mx::stack(all_hidden, -2), mx::stack(all_cell, -2)};
}

} // namespace rlbg::nn
