#include <mlx_util/containers.h>

#include <algorithm>

namespace rlbg::nn {

const ModulePtr& ModuleDict::operator[](const s::string& name) const {
  auto it = s::ranges::find_if(children(), [&](const NamedModule& c) {
    return c.first == name;
  });
  if (it == children().end())
    throw s::out_of_range("ModuleDict has no key " + name);
  return it->second;
}

Sequential& Sequential::append(UnaryModulePtr module) {
  layers->append(module);
  mLayers.push_back(s::move(module));
  return *this;
}

mx::array Sequential::forward(const mx::array& x) {
  mx::array y = x;
  for (const UnaryModulePtr& m : mLayers) y = m->forward(y);
  return y;
}

} // namespace rlbg::nn
