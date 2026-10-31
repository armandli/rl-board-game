#ifndef RLBG_MLX_UTIL_MODULE_H
#define RLBG_MLX_UTIL_MODULE_H

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <type_alias.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// flattened parameter tree: "layers.0.weight" -> array, the same keys
// Python's mlx.utils.tree_flatten(model.parameters()) produces
using Params = s::map<s::string, mx::array>;

// evaluates every array of a parameter map, like mx.eval(model.parameters())
inline void eval(const Params& params) {
  s::vector<mx::array> arrays;
  arrays.reserve(params.size());
  for (const auto& [k, v] : params) arrays.push_back(v);
  mx::eval(arrays);
}

struct Module;

using ModulePtr = s::shared_ptr<Module>;
using NamedModule = s::pair<s::string, ModulePtr>;
using ParamFilter =
  s::function<bool(const Module&, const s::string&, const mx::array&)>;
using ArrayFn = s::function<mx::array(const mx::array&)>;

// C++ counterpart of mlx.nn.Module. Parameters are data members of the
// derived module registered by reference, so modules are not copyable or
// movable; own them through s::shared_ptr.
struct Module {
  Module() = default;
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = delete;
  Module& operator=(Module&&) = delete;
  virtual ~Module() = default;

  virtual s::string type_name() const { return "Module"; }
  virtual s::string extra_repr() const { return ""; }
  // containers print their children as "(layers.0): ..." the way Python
  // prints a list member
  virtual bool is_container() const { return false; }

  bool training() const { return mTraining; }

  Params parameters() const;
  Params trainable_parameters() const;
  Params filter_and_map(
    const ParamFilter& filter,
    const ArrayFn& map = nullptr
  ) const;

  const s::vector<NamedModule>& children() const { return mChildren; }
  s::vector<s::pair<s::string, Module*>> named_modules();
  s::vector<Module*> modules();
  s::vector<s::pair<s::string, Module*>> leaf_modules();
  Module& apply_to_modules(
    const s::function<void(const s::string&, Module&)>& fn
  );

  bool has_parameter(s::string_view name) const;
  bool has_child(s::string_view name) const;

  Module& update(const Params& params, bool strict = true);
  Module& apply(const ArrayFn& map, const ParamFilter& filter = nullptr);
  Module& set_dtype(
    mx::Dtype dtype,
    const s::function<bool(mx::Dtype)>& predicate = nullptr
  );

  Module& load_weights(const s::string& file, bool strict = true);
  Module& load_weights(const Params& weights, bool strict = true);
  void save_weights(const s::string& file) const;

  // keys name parameters or child modules of each visited module; nullopt
  // means every parameter of that module
  Module& freeze(
    const s::optional<s::vector<s::string>>& keys = s::nullopt,
    bool recurse = true,
    bool strict = false
  );
  Module& unfreeze(
    const s::optional<s::vector<s::string>>& keys = s::nullopt,
    bool recurse = true,
    bool strict = false
  );

  Module& train(bool mode = true);
  Module& eval() { return train(false); }

protected:
  // value must be a data member of this module that outlives the
  // registration; a dotted name like "weights.0" matches Python arrays
  // stored inside a list attribute
  void register_parameter(s::string name, mx::array& value);
  template <typename M>
  s::shared_ptr<M> register_module(s::string name, s::shared_ptr<M> module);

  virtual void freeze_local(
    const s::optional<s::vector<s::string>>& keys,
    bool strict
  );
  virtual void unfreeze_local(
    const s::optional<s::vector<s::string>>& keys,
    bool strict
  );
  void validate_keys(const s::vector<s::string>& keys) const;
  bool is_frozen(const s::string& name) const;
  void collect(
    const s::string& prefix,
    const ParamFilter& filter,
    const ArrayFn& map,
    bool trainable_only,
    Params& out
  ) const;
  mx::array* find_parameter(s::string_view key);

private:
  s::vector<s::pair<s::string, mx::array*>> mParams;
  s::vector<NamedModule> mChildren;
  uset<s::string> mNoGrad;
  bool mTraining = true;
};

s::ostream& operator<<(s::ostream& os, const Module& m);

template <typename M>
s::shared_ptr<M> Module::register_module(
  s::string name,
  s::shared_ptr<M> module)
{
  mChildren.emplace_back(s::move(name), module);
  return module;
}

// modules with a single array input and output; required by Sequential
struct UnaryModule : Module {
  virtual mx::array forward(const mx::array& x) = 0;
  mx::array operator()(const mx::array& x) { return forward(x); }
};

using UnaryModulePtr = s::shared_ptr<UnaryModule>;

// computes fn() and its gradient with respect to model's trainable
// parameters. fn must read the parameters through model. The model's
// parameters are restored afterwards; gradients are keyed like
// trainable_parameters().
s::pair<mx::array, Params> compute_value_and_grad(
  Module& model,
  const s::function<mx::array()>& fn
);

// mlx.nn.value_and_grad(model, fn): returns a callable taking fn's arguments
// and returning {loss, grads}; model must outlive the returned callable
template <typename F>
auto value_and_grad(Module& model, F fn) {
  return [&model, fn = s::move(fn)](auto&&... args) {
    return compute_value_and_grad(model, [&]() { return fn(args...); });
  };
}

using ArraysFn =
  s::function<s::vector<mx::array>(const s::vector<mx::array>&)>;

// mlx.nn.checkpoint: evaluates fn(inputs), which may read module's
// parameters, without keeping its intermediate activations; the backward
// pass recomputes them. module must outlive any gradient computation that
// goes through the result.
s::vector<mx::array> checkpoint(
  Module& module,
  ArraysFn fn,
  const s::vector<mx::array>& inputs
);

// array scalar of x's dtype so mixing a C++ number with half precision
// arrays does not promote them to float32 the way an mx::array(float) does
inline mx::array scalar_like(double v, const mx::array& x) {
  mx::Dtype dt = mx::issubdtype(x.dtype(), mx::floating) ? x.dtype() : mx::float32;
  return mx::array(v, dt);
}

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_MODULE_H
