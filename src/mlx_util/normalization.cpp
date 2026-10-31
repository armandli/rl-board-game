#include <mlx_util/normalization.h>

#include <format>
#include <numeric>
#include <stdexcept>

#include <mlx_util/repr.h>

namespace rlbg::nn {

namespace {

void check_eps(const char* name, float eps) {
  if (eps <= 0.0f)
    throw s::invalid_argument(
      s::format("[{}] 'eps' must be positive but got {}.", name, py_float(eps))
    );
}

mx::array affine_or(
  const mx::array& x,
  const s::optional<mx::array>& weight,
  const s::optional<mx::array>& bias)
{
  return weight ? *weight * x + *bias : x;
}

} // namespace

InstanceNorm::InstanceNorm(int dims, float eps, bool affine)
  : dims(dims), eps(eps)
{
  check_eps("InstanceNorm", eps);
  if (affine) {
    weight = mx::ones({dims});
    bias = mx::zeros({dims});
    register_parameter("weight", *weight);
    register_parameter("bias", *bias);
  }
}

s::string InstanceNorm::extra_repr() const {
  return s::format(
    "{}, eps={}, affine={}", dims, py_float(eps), py_bool(weight.has_value())
  );
}

mx::array InstanceNorm::forward(const mx::array& x) {
  int nd = (int)x.ndim();
  if (nd < 3)
    throw s::invalid_argument(
      s::format(
        "InstanceNorm expects inputs with at least 3 dimensions (N, ..., C) "
        "but the input has {} dimensions.", nd
      )
    );
  int batch = x.shape(0);
  int features = x.shape(-1);
  mx::Shape spatial(x.shape().begin() + 1, x.shape().end() - 1);

  s::vector<int> to_channels_first{0, nd - 1};
  for (int i = 1; i < nd - 1; ++i) to_channels_first.push_back(i);
  mx::array y = mx::reshape(mx::transpose(x, to_channels_first), {batch, features, -1});
  y = mx::fast::layer_norm(y, s::nullopt, s::nullopt, eps);

  mx::Shape unflat{batch, features};
  unflat.insert(unflat.end(), spatial.begin(), spatial.end());
  s::vector<int> to_channels_last{0};
  for (int i = 2; i < (int)spatial.size() + 2; ++i) to_channels_last.push_back(i);
  to_channels_last.push_back(1);
  y = mx::transpose(mx::reshape(y, unflat), to_channels_last);
  return affine_or(y, weight, bias);
}

LayerNorm::LayerNorm(int dims, float eps, bool affine, bool bias)
  : eps(eps), dims(dims)
{
  check_eps("LayerNorm", eps);
  if (affine) {
    weight = mx::ones({dims});
    register_parameter("weight", *weight);
    if (bias) {
      this->bias = mx::zeros({dims});
      register_parameter("bias", *this->bias);
    }
  }
}

s::string LayerNorm::extra_repr() const {
  return s::format(
    "{}, eps={}, affine={}", dims, py_float(eps), py_bool(weight.has_value())
  );
}

RMSNorm::RMSNorm(int dims, float eps) : weight(mx::ones({dims})), eps(eps) {
  check_eps("RMSNorm", eps);
  register_parameter("weight", weight);
}

s::string RMSNorm::extra_repr() const {
  return s::format("{}, eps={}", weight.shape(0), py_float(eps));
}

GroupNorm::GroupNorm(
  int num_groups,
  int dims,
  float eps,
  bool affine,
  bool pytorch_compatible)
  : num_groups(num_groups),
    dims(dims),
    eps(eps),
    pytorch_compatible(pytorch_compatible)
{
  check_eps("GroupNorm", eps);
  if (num_groups <= 0)
    throw s::invalid_argument(s::format(
      "The number of groups ({}) must be a positive integer.", num_groups
    ));
  if (dims <= 0)
    throw s::invalid_argument(s::format(
      "The number of features ({}) must be a positive integer.", dims
    ));
  if (dims % num_groups != 0)
    throw s::invalid_argument(s::format(
      "The number of features ({}) must be evenly divisible by the number of "
      "groups ({}).", dims, num_groups
    ));
  if (affine) {
    bias = mx::zeros({dims});
    weight = mx::ones({dims});
    register_parameter("bias", *bias);
    register_parameter("weight", *weight);
  }
}

s::string GroupNorm::extra_repr() const {
  return s::format(
    "{}, {}, eps={}, affine={}, pytorch_compatible={}",
    num_groups, dims, py_float(eps), py_bool(weight.has_value()),
    py_bool(pytorch_compatible)
  );
}

mx::array GroupNorm::pytorch_compatible_group_norm(const mx::array& x) const {
  int batch = x.shape(0);
  int group_size = x.shape(-1) / num_groups;
  mx::array y = mx::reshape(x, {batch, -1, num_groups, group_size});
  y = mx::reshape(mx::transpose(y, {0, 2, 1, 3}), {batch, num_groups, -1});
  y = mx::fast::layer_norm(y, s::nullopt, s::nullopt, eps);
  y = mx::reshape(y, {batch, num_groups, -1, group_size});
  return mx::reshape(mx::transpose(y, {0, 2, 1, 3}), x.shape());
}

mx::array GroupNorm::group_norm(const mx::array& x) const {
  mx::array y = mx::reshape(x, {x.shape(0), -1, num_groups});
  mx::array means = mx::mean(y, 1, true);
  mx::array var = mx::var(y, 1, true);
  y = (y - means) * mx::rsqrt(var + scalar_like(eps, var));
  return mx::reshape(y, x.shape());
}

mx::array GroupNorm::forward(const mx::array& x) {
  mx::array y = pytorch_compatible ? pytorch_compatible_group_norm(x) : group_norm(x);
  return affine_or(y, weight, bias);
}

BatchNorm::BatchNorm(
  int num_features,
  float eps,
  double momentum,
  bool affine,
  bool track_running_stats)
  : num_features(num_features),
    eps(eps),
    momentum(momentum),
    track_running_stats(track_running_stats)
{
  check_eps("BatchNorm", eps);
  if (affine) {
    weight = mx::ones({num_features});
    bias = mx::zeros({num_features});
    register_parameter("weight", *weight);
    register_parameter("bias", *bias);
  }
  if (track_running_stats) {
    running_mean = mx::zeros({num_features});
    running_var = mx::ones({num_features});
    register_parameter("running_mean", *running_mean);
    register_parameter("running_var", *running_var);
    freeze_running_stats();
  }
}

void BatchNorm::freeze_running_stats() {
  if (track_running_stats)
    Module::freeze_local(s::vector<s::string>{"running_mean", "running_var"}, false);
}

// Python's BatchNorm only re-freezes the running stats when unfreeze() is
// called on the BatchNorm itself; doing it per module keeps them frozen
// when a parent model is unfrozen too, so optimizers never touch them
void BatchNorm::unfreeze_local(
  const s::optional<s::vector<s::string>>& keys,
  bool strict)
{
  Module::unfreeze_local(keys, strict);
  freeze_running_stats();
}

s::string BatchNorm::extra_repr() const {
  return s::format(
    "{}, eps={}, momentum={}, affine={}, track_running_stats={}",
    num_features, py_float(eps), py_float(momentum),
    py_bool(weight.has_value()), py_bool(track_running_stats)
  );
}

mx::array BatchNorm::forward(const mx::array& x) {
  int nd = (int)x.ndim();
  if (nd < 2 or nd > 4)
    throw s::invalid_argument(s::format(
      "Expected input tensor to have 2, 3 or 4 dimensions, but got {}", nd
    ));
  if (training()) {
    size_t stats_size = x.size() / x.shape(-1);
    if (stats_size == 1)
      throw s::invalid_argument(
        "BatchNorm training requires more than one value per channel."
      );
  }
  s::vector<int> axes(nd - 1);
  s::iota(axes.begin(), axes.end(), 0);
  mx::array mean = mx::mean(x, axes);
  mx::array var = mx::var(x, axes);

  if (training() and track_running_stats) {
    mx::array mu = scalar_like(momentum, *running_mean);
    mx::array one_minus_mu = scalar_like(1.0 - momentum, *running_mean);
    mx::array unbiased_var = mx::var(x, axes, false, 1);
    *running_mean = one_minus_mu * *running_mean + mu * mean;
    *running_var = one_minus_mu * *running_var + mu * unbiased_var;
  } else if (track_running_stats) {
    mean = *running_mean;
    var = *running_var;
  }
  mx::array y = (x - mean) * mx::rsqrt(var + scalar_like(eps, var));
  return affine_or(y, weight, bias);
}

} // namespace rlbg::nn
