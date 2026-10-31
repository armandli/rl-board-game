#ifndef RLBG_MLX_UTIL_REPR_H
#define RLBG_MLX_UTIL_REPR_H

#include <format>
#include <string>
#include <vector>

namespace rlbg::nn {

namespace s = std;

// Python repr() spellings so module printouts read like mlx.nn's

inline s::string py_bool(bool v) { return v ? "True" : "False"; }

inline s::string py_float(double v) {
  s::string out = s::format("{}", v);
  if (out.find_first_of(".en") == s::string::npos) out += ".0";
  return out;
}

template <typename T>
s::string py_tuple(const s::vector<T>& v) {
  s::string out = "(";
  for (size_t i = 0; i < v.size(); ++i)
    out += (i ? ", " : "") + s::format("{}", v[i]);
  return out + (v.size() == 1 ? ",)" : ")");
}

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_REPR_H
