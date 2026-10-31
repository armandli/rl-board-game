#include <mlx_util/module.h>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <mlx_util/io.h>

namespace rlbg::nn {

namespace {

bool is_dotted_prefix(s::string_view prefix, s::string_view name) {
  return name.size() > prefix.size() and name.starts_with(prefix) and
         name[prefix.size()] == '.';
}

s::string join_key(const s::string& prefix, const s::string& name) {
  return prefix.empty() ? name : prefix + "." + name;
}

void walk_modules(
  const s::string& prefix,
  Module& m,
  const s::function<void(const s::string&, Module&)>& fn)
{
  fn(prefix, m);
  for (const auto& [name, child] : m.children())
    walk_modules(join_key(prefix, name), *child, fn);
}

void flatten_children(
  const s::string& prefix,
  const Module& m,
  s::vector<s::pair<s::string, const Module*>>& out)
{
  for (const auto& [name, child] : m.children()) {
    s::string key = join_key(prefix, name);
    if (child->is_container()) flatten_children(key, *child, out);
    else                       out.emplace_back(key, child.get());
  }
}

s::string indent_lines(const s::string& text, const s::string& pad) {
  s::string out;
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find('\n', begin);
    if (end == s::string::npos) end = text.size();
    out += pad + text.substr(begin, end - begin);
    if (end == text.size()) break;
    out += '\n';
    begin = end + 1;
  }
  return out;
}

s::string repr_string(const Module& m) {
  s::string out = m.type_name() + "(" + m.extra_repr();
  s::vector<s::pair<s::string, const Module*>> kids;
  flatten_children("", m, kids);
  for (const auto& [key, child] : kids)
    out += "\n" + indent_lines("(" + key + "): " + repr_string(*child), "  ");
  if (not kids.empty()) out += "\n";
  return out + ")";
}

s::string shape_string(const mx::Shape& shape) {
  s::ostringstream os;
  os << "(";
  for (size_t i = 0; i < shape.size(); ++i)
    os << (i ? ", " : "") << shape[i];
  os << ")";
  return os.str();
}

} // namespace

void Module::register_parameter(s::string name, mx::array& value) {
  for (auto& [n, p] : mParams)
    if (n == name) {
      p = &value;
      return;
    }
  mParams.emplace_back(s::move(name), &value);
}

bool Module::is_frozen(const s::string& name) const {
  if (mNoGrad.contains(name)) return true;
  for (const s::string& k : mNoGrad)
    if (is_dotted_prefix(k, name)) return true;
  return false;
}

void Module::collect(
  const s::string& prefix,
  const ParamFilter& filter,
  const ArrayFn& map,
  bool trainable_only,
  Params& out) const
{
  for (const auto& [name, value] : mParams) {
    if (trainable_only and is_frozen(name)) continue;
    if (filter and not filter(*this, name, *value)) continue;
    out.insert_or_assign(prefix + name, map ? map(*value) : *value);
  }
  for (const auto& [name, child] : mChildren) {
    if (trainable_only and is_frozen(name)) continue;
    child->collect(prefix + name + ".", filter, map, trainable_only, out);
  }
}

Params Module::parameters() const {
  Params out;
  collect("", nullptr, nullptr, false, out);
  return out;
}

Params Module::trainable_parameters() const {
  Params out;
  collect("", nullptr, nullptr, true, out);
  return out;
}

Params Module::filter_and_map(
  const ParamFilter& filter,
  const ArrayFn& map) const
{
  Params out;
  collect("", filter, map, false, out);
  return out;
}

Module& Module::apply_to_modules(
  const s::function<void(const s::string&, Module&)>& fn)
{
  walk_modules("", *this, fn);
  return *this;
}

s::vector<s::pair<s::string, Module*>> Module::named_modules() {
  s::vector<s::pair<s::string, Module*>> out;
  apply_to_modules([&](const s::string& k, Module& m) {
    out.emplace_back(k, &m);
  });
  return out;
}

s::vector<Module*> Module::modules() {
  s::vector<Module*> out;
  apply_to_modules([&](const s::string&, Module& m) { out.push_back(&m); });
  return out;
}

s::vector<s::pair<s::string, Module*>> Module::leaf_modules() {
  s::vector<s::pair<s::string, Module*>> out;
  apply_to_modules([&](const s::string& k, Module& m) {
    if (m.children().empty() and &m != this) out.emplace_back(k, &m);
  });
  return out;
}

bool Module::has_parameter(s::string_view name) const {
  return s::ranges::any_of(mParams, [&](const auto& p) {
    return p.first == name or is_dotted_prefix(name, p.first);
  });
}

bool Module::has_child(s::string_view name) const {
  return s::ranges::any_of(mChildren, [&](const NamedModule& c) {
    return c.first == name;
  });
}

mx::array* Module::find_parameter(s::string_view key) {
  for (auto& [name, value] : mParams)
    if (name == key) return value;
  for (auto& [name, child] : mChildren)
    if (is_dotted_prefix(name, key))
      if (mx::array* p = child->find_parameter(key.substr(name.size() + 1)))
        return p;
  return nullptr;
}

Module& Module::update(const Params& params, bool strict) {
  for (const auto& [key, value] : params) {
    mx::array* p = find_parameter(key);
    if (p == nullptr) {
      if (strict)
        throw s::invalid_argument(
          "Module does not have parameter named \"" + key + "\"."
        );
      continue;
    }
    *p = value;
  }
  return *this;
}

Module& Module::apply(const ArrayFn& map, const ParamFilter& filter) {
  return update(filter_and_map(filter, map));
}

Module& Module::set_dtype(
  mx::Dtype dtype,
  const s::function<bool(mx::Dtype)>& predicate)
{
  return apply([&](const mx::array& x) {
    bool convert = predicate ? predicate(x.dtype())
                             : mx::issubdtype(x.dtype(), mx::floating);
    return convert ? mx::astype(x, dtype) : x;
  });
}

Module& Module::load_weights(const s::string& file, bool strict) {
  return load_weights(load_arrays(file), strict);
}

Module& Module::load_weights(const Params& weights, bool strict) {
  if (strict) {
    Params current = parameters();
    s::vector<s::string> extras;
    s::vector<s::string> missing;
    for (const auto& [k, v] : weights)
      if (not current.contains(k)) extras.push_back(k);
    for (const auto& [k, v] : current)
      if (not weights.contains(k)) missing.push_back(k);
    auto join = [](const s::vector<s::string>& keys) {
      s::string out;
      for (const s::string& k : keys) out += (out.empty() ? "" : ",\n") + k;
      return out;
    };
    if (not extras.empty())
      throw s::invalid_argument(
        "Received " + s::to_string(extras.size()) +
        " parameters not in model: \n" + join(extras) + "."
      );
    if (not missing.empty())
      throw s::invalid_argument(
        "Missing " + s::to_string(missing.size()) + " parameters: \n" +
        join(missing) + "."
      );
    for (const auto& [k, v] : current) {
      const mx::array& nv = weights.at(k);
      if (nv.shape() != v.shape())
        throw s::invalid_argument(
          "Expected shape " + shape_string(v.shape()) +
          " but received shape " + shape_string(nv.shape()) +
          " for parameter " + k
        );
    }
  }
  if (not weights.empty()) update(weights, false);
  return *this;
}

void Module::save_weights(const s::string& file) const {
  save_arrays(file, parameters());
}

void Module::validate_keys(const s::vector<s::string>& keys) const {
  for (const s::string& k : keys)
    if (not has_parameter(k) and not has_child(k))
      throw s::out_of_range("Module doesn't contain member " + k + ".");
}

void Module::freeze_local(
  const s::optional<s::vector<s::string>>& keys,
  bool strict)
{
  if (not keys) {
    for (const auto& [name, value] : mParams) mNoGrad.insert(name);
    return;
  }
  if (strict) validate_keys(*keys);
  mNoGrad.insert(keys->begin(), keys->end());
}

void Module::unfreeze_local(
  const s::optional<s::vector<s::string>>& keys,
  bool strict)
{
  if (not keys) {
    mNoGrad.clear();
    return;
  }
  if (strict) validate_keys(*keys);
  for (const s::string& k : *keys) mNoGrad.erase(k);
}

Module& Module::freeze(
  const s::optional<s::vector<s::string>>& keys,
  bool recurse,
  bool strict)
{
  if (not recurse) {
    freeze_local(keys, strict);
    return *this;
  }
  if (strict and keys) {
    s::vector<Module*> all = modules();
    for (const s::string& k : *keys)
      if (s::ranges::none_of(all, [&](const Module* m) {
            return m->has_parameter(k) or m->has_child(k);
          }))
        throw s::out_of_range("Module doesn't contain member " + k + ".");
  }
  return apply_to_modules([&](const s::string&, Module& m) {
    m.freeze_local(keys, false);
  });
}

Module& Module::unfreeze(
  const s::optional<s::vector<s::string>>& keys,
  bool recurse,
  bool strict)
{
  if (not recurse) {
    unfreeze_local(keys, strict);
    return *this;
  }
  if (strict and keys) {
    s::vector<Module*> all = modules();
    for (const s::string& k : *keys)
      if (s::ranges::none_of(all, [&](const Module* m) {
            return m->has_parameter(k) or m->has_child(k);
          }))
        throw s::out_of_range("Module doesn't contain member " + k + ".");
  }
  return apply_to_modules([&](const s::string&, Module& m) {
    m.unfreeze_local(keys, false);
  });
}

Module& Module::train(bool mode) {
  return apply_to_modules([mode](const s::string&, Module& m) {
    m.mTraining = mode;
  });
}

s::ostream& operator<<(s::ostream& os, const Module& m) {
  return os << repr_string(m);
}

s::pair<mx::array, Params> compute_value_and_grad(
  Module& model,
  const s::function<mx::array()>& fn)
{
  const Params original = model.trainable_parameters();
  if (original.empty()) return {fn(), Params{}};

  s::vector<s::string> keys;
  s::vector<mx::array> values;
  for (const auto& [k, v] : original) {
    keys.push_back(k);
    values.push_back(v);
  }
  s::vector<int> argnums(values.size());
  s::iota(argnums.begin(), argnums.end(), 0);

  s::function<mx::array(const s::vector<mx::array>&)> inner =
    [&](const s::vector<mx::array>& xs) {
      Params traced;
      for (size_t i = 0; i < xs.size(); ++i) traced.insert_or_assign(keys[i], xs[i]);
      model.update(traced);
      return fn();
    };

  // restore the untraced parameters even if fn throws, so the model never
  // keeps arrays that belong to the gradient trace
  struct Restore {
    ~Restore() { model.update(original); }
    Module& model;
    const Params& original;
  } restore{model, original};

  auto [value, grads] = mx::value_and_grad(inner, argnums)(values);
  Params out;
  for (size_t i = 0; i < keys.size(); ++i) out.insert_or_assign(keys[i], grads[i]);
  return {value, out};
}

s::vector<mx::array> checkpoint(
  Module& module,
  ArraysFn fn,
  const s::vector<mx::array>& inputs)
{
  const Params params = module.trainable_parameters();
  s::vector<s::string> keys;
  s::vector<mx::array> all_inputs;
  for (const auto& [k, v] : params) {
    keys.push_back(k);
    all_inputs.push_back(v);
  }
  all_inputs.insert(all_inputs.end(), inputs.begin(), inputs.end());

  // mx::checkpoint keeps this closure to rerun it in the backward pass, so
  // it captures by value and puts the module's parameters back each time
  Module* m = &module;
  ArraysFn inner = [m, keys, fn = s::move(fn)](const s::vector<mx::array>& xs) {
    Params saved = m->trainable_parameters();
    Params traced;
    for (size_t i = 0; i < keys.size(); ++i) traced.insert_or_assign(keys[i], xs[i]);
    m->update(traced);
    s::vector<mx::array> args(xs.begin() + (long)keys.size(), xs.end());
    s::vector<mx::array> out = fn(args);
    m->update(saved);
    return out;
  };
  return mx::checkpoint(inner)(all_inputs);
}

} // namespace rlbg::nn
