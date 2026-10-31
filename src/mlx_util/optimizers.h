#ifndef RLBG_MLX_UTIL_OPTIMIZERS_H
#define RLBG_MLX_UTIL_OPTIMIZERS_H

#include <concepts>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/module.h>
#include <mlx_util/schedulers.h>

namespace rlbg::optim {

namespace s = std;
namespace mx = mlx::core;

using nn::Module;
using nn::Params;

// a fixed hyperparameter value or a Schedule of the step counter
struct Hyper {
  Hyper(double value) : value(value) {}
  template <typename F>
    requires s::invocable<F, const mx::array&>
  Hyper(F schedule) : schedule(s::move(schedule)) {}

  double value = 0.0;
  Schedule schedule;
};

// mlx.optimizers.Optimizer. state() flattens the optimizer state with the
// keys Python's tree_flatten(optimizer.state) produces ("step",
// "learning_rate", "<parameter>.<slot>"), so files written by save_state()
// load into a Python optimizer through
//   optimizer.state = tree_unflatten(list(mx.load(file).items()))
// and files saved from Python with
//   mx.save_safetensors(file, dict(tree_flatten(optimizer.state)))
// load here with load_state().
struct Optimizer {
  using Slots = s::map<s::string, mx::array>;

  Optimizer();
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;
  virtual ~Optimizer() = default;

  void update(Module& model, const Params& gradients) {
    model.update(apply_gradients(gradients, model.parameters()));
  }
  // parameters must hold every key of gradients; returns the updated values
  // of the gradients' keys
  virtual Params apply_gradients(const Params& gradients, const Params& parameters);
  virtual void init(const Params& parameters);

  virtual Params state() const;
  virtual void set_state(const Params& state);
  void save_state(const s::string& file) const;
  void load_state(const s::string& file);

  mx::array step() const { return mScalars.at("step"); }
  virtual mx::array learning_rate() const { return mScalars.at("learning_rate"); }
  virtual void set_learning_rate(double learning_rate) {
    mScalars.insert_or_assign("learning_rate", mx::array((float)learning_rate));
  }

protected:
  virtual void init_single(const mx::array& parameter, Slots& state) = 0;
  virtual mx::array apply_single(
    const mx::array& gradient,
    const mx::array& parameter,
    Slots& state
  ) = 0;
  void maybe_schedule(const s::string& name, const Hyper& value);
  // the hyperparameter in the gradient's dtype, as Python's
  // self.learning_rate.astype(gradient.dtype)
  mx::array lr_like(const mx::array& gradient) const {
    return mx::astype(learning_rate(), gradient.dtype());
  }

  // scalars at the top of Python's state dict: "step", "learning_rate" and
  // any scheduled hyperparameter
  Params mScalars;
  s::map<s::string, Slots> mSlots;
  s::map<s::string, Schedule> mSchedulers;
  bool mInitialized = false;
};

struct SGD : Optimizer {
  explicit SGD(
    Hyper learning_rate,
    double momentum = 0.0,
    double weight_decay = 0.0,
    double dampening = 0.0,
    bool nesterov = false
  );

  double momentum;
  double weight_decay;
  double dampening;
  bool nesterov;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct RMSprop : Optimizer {
  explicit RMSprop(Hyper learning_rate, double alpha = 0.99, double eps = 1e-8);

  double alpha;
  double eps;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct Adagrad : Optimizer {
  explicit Adagrad(Hyper learning_rate, double eps = 1e-8);

  double eps;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct AdaDelta : Optimizer {
  explicit AdaDelta(Hyper learning_rate, double rho = 0.9, double eps = 1e-6);

  double rho;
  double eps;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct Adam : Optimizer {
  explicit Adam(
    Hyper learning_rate,
    s::pair<double, double> betas = {0.9, 0.999},
    double eps = 1e-8,
    bool bias_correction = false
  );

  s::pair<double, double> betas;
  double eps;
  bool bias_correction;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct AdamW : Adam {
  explicit AdamW(
    Hyper learning_rate,
    s::pair<double, double> betas = {0.9, 0.999},
    double eps = 1e-8,
    double weight_decay = 0.01,
    bool bias_correction = false
  );

  double weight_decay;

protected:
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct Adamax : Adam {
  explicit Adamax(
    Hyper learning_rate,
    s::pair<double, double> betas = {0.9, 0.999},
    double eps = 1e-8
  );

protected:
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct Lion : Optimizer {
  explicit Lion(
    Hyper learning_rate,
    s::pair<double, double> betas = {0.9, 0.99},
    double weight_decay = 0.0
  );

  s::pair<double, double> betas;
  double weight_decay;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

struct Adafactor : Optimizer {
  explicit Adafactor(
    s::optional<Hyper> learning_rate = s::nullopt,
    s::pair<double, double> eps = {1e-30, 1e-3},
    double clip_threshold = 1.0,
    double decay_rate = -0.8,
    s::optional<double> beta_1 = s::nullopt,
    double weight_decay = 0.0,
    bool scale_parameter = true,
    bool relative_step = true,
    bool warmup_init = false
  );

  s::pair<double, double> eps;
  double clip_threshold;
  double decay_rate;
  s::optional<double> beta_1;
  double weight_decay;
  bool scale_parameter;
  bool relative_step;
  bool warmup_init;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
  mx::array compute_learning_rate(const mx::array& step, const mx::array& parameter_rms) const;
};

struct Muon : Optimizer {
  explicit Muon(
    Hyper learning_rate,
    double momentum = 0.95,
    double weight_decay = 0.01,
    bool nesterov = true,
    int ns_steps = 5
  );

  double momentum;
  double weight_decay;
  bool nesterov;
  int ns_steps;

protected:
  void init_single(const mx::array& parameter, Slots& state) override;
  mx::array apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) override;
};

// routes each gradient to the first optimizer whose filter accepts its key;
// the last optimizer takes everything else, so filters has one entry fewer
// than optimizers
struct MultiOptimizer : Optimizer {
  using Filter = s::function<bool(const s::string& key, const mx::array& gradient)>;

  MultiOptimizer(s::vector<s::shared_ptr<Optimizer>> optimizers, s::vector<Filter> filters = {});

  Params apply_gradients(const Params& gradients, const Params& parameters) override;
  void init(const Params& parameters) override;
  // keys are "states.<i>.<key of optimizer i>", as in Python
  Params state() const override;
  void set_state(const Params& state) override;
  mx::array learning_rate() const override { return mOptimizers.at(0)->learning_rate(); }
  void set_learning_rate(double learning_rate) override;

protected:
  s::vector<Params> split(const Params& gradients) const;
  void init_single(const mx::array&, Slots&) override {}
  mx::array apply_single(const mx::array&, const mx::array& parameter, Slots&) override {
    return parameter;
  }

  s::vector<s::shared_ptr<Optimizer>> mOptimizers;
  s::vector<Filter> mFilters;
};

// scales gradients so their global L2 norm is at most max_norm; returns
// {clipped gradients, norm before clipping}
s::pair<Params, mx::array> clip_grad_norm(const Params& gradients, double max_norm);

} // namespace rlbg::optim

#endif//RLBG_MLX_UTIL_OPTIMIZERS_H
