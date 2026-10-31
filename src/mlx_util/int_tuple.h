#ifndef RLBG_MLX_UTIL_INT_TUPLE_H
#define RLBG_MLX_UTIL_INT_TUPLE_H

#include <algorithm>
#include <array>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mlx_util/repr.h>

namespace rlbg::nn {

namespace s = std;

// accepts an int or a braced list of N ints, the way Python layers take
// "int or tuple" arguments: Conv2d(3, 8, 3) or Conv2d(3, 8, {3, 5}).
// A default constructed IntN stands for Python's None.
template <size_t N>
struct IntN {
  IntN() : given(false) { values.fill(0); }
  IntN(int v) { values.fill(v); }
  IntN(s::initializer_list<int> list) {
    if (list.size() != N)
      throw s::invalid_argument(
        "expected an integer or a tuple of " + s::to_string(N) + " integers"
      );
    s::copy(list.begin(), list.end(), values.begin());
  }

  int operator[](size_t i) const { return values[i]; }
  s::pair<int, int> pair() const requires (N == 2) { return {values[0], values[1]}; }
  s::tuple<int, int, int> tuple() const requires (N == 3) {
    return {values[0], values[1], values[2]};
  }
  s::vector<int> vector() const { return {values.begin(), values.end()}; }
  s::string repr() const { return py_tuple(vector()); }

  s::array<int, N> values;
  bool given = true;
};

using Int1 = IntN<1>;
using Int2 = IntN<2>;
using Int3 = IntN<3>;

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_INT_TUPLE_H
