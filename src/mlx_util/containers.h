#ifndef RLBG_MLX_UTIL_CONTAINERS_H
#define RLBG_MLX_UTIL_CONTAINERS_H

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// wraps a plain function so it can sit in a Sequential; it has no
// parameters but still takes up a list index, as a Python function does
struct Lambda : UnaryModule {
  explicit Lambda(ArrayFn fn, s::string name = "Lambda")
    : fn(s::move(fn)), mName(s::move(name)) {}

  s::string type_name() const override { return mName; }
  mx::array forward(const mx::array& x) override { return fn(x); }

  ArrayFn fn;

private:
  s::string mName;
};

// a Python list of modules: children are keyed "0", "1", ...
struct ModuleList : Module {
  s::string type_name() const override { return "ModuleList"; }
  bool is_container() const override { return true; }

  template <typename M>
  s::shared_ptr<M> append(s::shared_ptr<M> module) {
    return register_module(s::to_string(children().size()), s::move(module));
  }
  size_t size() const { return children().size(); }
  const ModulePtr& operator[](size_t i) const { return children().at(i).second; }
  template <typename M>
  s::shared_ptr<M> at(size_t i) const {
    auto m = s::dynamic_pointer_cast<M>(children().at(i).second);
    if (not m) throw s::bad_cast();
    return m;
  }
};

// a Python dict of modules: children are keyed by name
struct ModuleDict : Module {
  s::string type_name() const override { return "ModuleDict"; }
  bool is_container() const override { return true; }

  template <typename M>
  s::shared_ptr<M> insert(s::string name, s::shared_ptr<M> module) {
    if (has_child(name))
      throw s::invalid_argument("ModuleDict already has key " + name);
    return register_module(s::move(name), s::move(module));
  }
  bool contains(const s::string& name) const { return has_child(name); }
  const ModulePtr& operator[](const s::string& name) const;
  template <typename M>
  s::shared_ptr<M> at(const s::string& name) const {
    auto m = s::dynamic_pointer_cast<M>((*this)[name]);
    if (not m) throw s::bad_cast();
    return m;
  }
};

struct Sequential : UnaryModule {
  template <typename... Ts>
  explicit Sequential(Ts&&... items)
    : layers(register_module("layers", s::make_shared<ModuleList>())) {
    (append(s::forward<Ts>(items)), ...);
  }

  s::string type_name() const override { return "Sequential"; }
  mx::array forward(const mx::array& x) override;

  Sequential& append(UnaryModulePtr module);
  Sequential& append(ArrayFn fn) {
    return append(s::make_shared<Lambda>(s::move(fn)));
  }
  size_t size() const { return mLayers.size(); }
  const UnaryModulePtr& operator[](size_t i) const { return mLayers.at(i); }

  s::shared_ptr<ModuleList> layers;

private:
  s::vector<UnaryModulePtr> mLayers;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_CONTAINERS_H
