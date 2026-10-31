#ifndef RLBG_MLX_UTIL_NORMALIZATION_H
#define RLBG_MLX_UTIL_NORMALIZATION_H

#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// normalizes each channel of (N, ..., C) inputs over the spatial axes
struct InstanceNorm : UnaryModule {
  explicit InstanceNorm(int dims, float eps = 1e-5f, bool affine = false);

  s::string type_name() const override { return "InstanceNorm"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  s::optional<mx::array> weight;
  s::optional<mx::array> bias;
  int dims;
  float eps;
};

struct LayerNorm : UnaryModule {
  explicit LayerNorm(int dims, float eps = 1e-5f, bool affine = true, bool bias = true);

  s::string type_name() const override { return "LayerNorm"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override {
    return mx::fast::layer_norm(x, weight, bias, eps);
  }

  s::optional<mx::array> weight;
  s::optional<mx::array> bias;
  float eps;
  int dims;
};

struct RMSNorm : UnaryModule {
  explicit RMSNorm(int dims, float eps = 1e-5f);

  s::string type_name() const override { return "RMSNorm"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override {
    return mx::fast::rms_norm(x, weight, eps);
  }

  mx::array weight;
  float eps;
};

struct GroupNorm : UnaryModule {
  GroupNorm(
    int num_groups,
    int dims,
    float eps = 1e-5f,
    bool affine = true,
    bool pytorch_compatible = false
  );

  s::string type_name() const override { return "GroupNorm"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  s::optional<mx::array> bias;
  s::optional<mx::array> weight;
  int num_groups;
  int dims;
  float eps;
  bool pytorch_compatible;

protected:
  mx::array pytorch_compatible_group_norm(const mx::array& x) const;
  mx::array group_norm(const mx::array& x) const;
};

// running_mean and running_var are saved with the weights but stay frozen
struct BatchNorm : UnaryModule {
  explicit BatchNorm(
    int num_features,
    float eps = 1e-5f,
    double momentum = 0.1,
    bool affine = true,
    bool track_running_stats = true
  );

  s::string type_name() const override { return "BatchNorm"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override;

  s::optional<mx::array> weight;
  s::optional<mx::array> bias;
  s::optional<mx::array> running_mean;
  s::optional<mx::array> running_var;
  int num_features;
  float eps;
  double momentum;
  bool track_running_stats;

protected:
  void unfreeze_local(
    const s::optional<s::vector<s::string>>& keys,
    bool strict
  ) override;
  void freeze_running_stats();
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_NORMALIZATION_H
