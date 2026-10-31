#include <mlx_util/dropout.h>

#include <stdexcept>

#include <mlx_util/repr.h>

namespace rlbg::nn {

Dropout::Dropout(double p) : mP1(1.0 - p) {
  if (p < 0 or p >= 1)
    throw s::invalid_argument(
      "The dropout probability " + py_float(p) + " is not in [0, 1)"
    );
}

s::string Dropout::extra_repr() const { return "p=" + py_float(1.0 - mP1); }

mx::array Dropout::apply_mask(const mx::array& x, int shared_axes) const {
  if (mP1 == 1.0 or not training()) return x;
  mx::Shape mask_shape = x.shape();
  for (int i = 2; i < 2 + shared_axes; ++i) mask_shape[mask_shape.size() - i] = 1;
  mx::array mask = mx::random::bernoulli(mx::array((float)mP1), mask_shape);
  return (mask * x) * scalar_like(1.0 / mP1, x);
}

mx::array Dropout::forward(const mx::array& x) { return apply_mask(x, 0); }

mx::array Dropout2d::forward(const mx::array& x) {
  if (x.ndim() != 3 and x.ndim() != 4)
    throw s::invalid_argument(
      "Received input with " + s::to_string(x.ndim()) +
      " dimensions. Expected 3 or 4 dimensions."
    );
  return apply_mask(x, 2);
}

mx::array Dropout3d::forward(const mx::array& x) {
  if (x.ndim() != 4 and x.ndim() != 5)
    throw s::invalid_argument(
      "Received input with " + s::to_string(x.ndim()) +
      " dimensions. Expected 4 or 5 dimensions."
    );
  return apply_mask(x, 3);
}

} // namespace rlbg::nn
