#include <mlx_util/losses.h>

#include <cassert>
#include <cmath>

#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace {

mx::array reduce(const mx::array& loss, Reduction reduction) {
  switch (reduction) {
  case Reduction::Mean:
    return mx::mean(loss);

  break; case Reduction::Sum:
    return mx::sum(loss);

  break; case Reduction::None:
    return loss;

  break; default:
    assert(false); // should never get here
    return loss;
  }
}

s::string shape_str(const mx::array& a) {
  s::ostringstream os;
  os << "(";
  for (size_t i = 0; i < a.ndim(); ++i) os << (i ? ", " : "") << a.shape(i);
  os << (a.ndim() == 1 ? ",)" : ")");
  return os.str();
}

void check_same_shape(const char* what, const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape())
    throw s::invalid_argument(
      s::string(what) + " shape " + shape_str(a) +
      " does not match targets shape " + shape_str(b) + "."
    );
}

void apply_weights(mx::array& loss, const s::optional<mx::array>& weights) {
  if (not weights) return;
  if (weights->shape() != loss.shape())
    throw s::invalid_argument(
      "Weights with shape " + shape_str(*weights) +
      " is not the same as output loss with shape " + shape_str(loss) + "."
    );
  loss = loss * *weights;
}

} // namespace

mx::array cross_entropy(
  const mx::array& logits,
  const mx::array& targets,
  const s::optional<mx::array>& weights,
  int axis,
  double label_smoothing,
  Reduction reduction)
{
  if (label_smoothing < 0 or label_smoothing >= 1)
    throw s::invalid_argument("Label smoothing must in [0, 1).");
  int nd = (int)logits.ndim();
  int ax = axis < 0 ? axis + nd : axis;
  bool targets_as_probs = targets.ndim() == logits.ndim();
  mx::Shape dropped = logits.shape();
  dropped.erase(dropped.begin() + ax);
  if ((targets_as_probs and targets.shape() != logits.shape()) or
      (not targets_as_probs and targets.shape() != dropped))
    throw s::invalid_argument(
      "Targets shape " + shape_str(targets) + " does not match logits shape " +
      shape_str(logits) + "."
    );

  bool use_fast = mx::default_device() == mx::Device(mx::Device::gpu) and
                  not targets_as_probs and label_smoothing == 0 and
                  ax == nd - 1 and
                  mx::issubdtype(logits.dtype(), mx::floating) and
                  mx::issubdtype(targets.dtype(), mx::integer);
  mx::array loss = logits;
  if (use_fast) {
    loss = mx::astype(mx::fast::cross_entropy(logits, targets), logits.dtype());
  } else {
    mx::array l = logits - mx::stop_gradient(mx::max(logits, ax, true));
    mx::array score = targets_as_probs
      ? mx::sum(l * targets, ax)
      : mx::squeeze(mx::take_along_axis(l, mx::expand_dims(targets, ax), ax), ax);
    mx::array lse = mx::logsumexp(l, ax);
    if (label_smoothing > 0) {
      mx::array adjusted = scalar_like(1 - label_smoothing, score) * score;
      mx::array smoothed = -mx::mean(l, ax) * scalar_like(label_smoothing, l);
      loss = lse - adjusted + smoothed;
    } else {
      loss = lse - score;
    }
  }
  apply_weights(loss, weights);
  return reduce(loss, reduction);
}

mx::array binary_cross_entropy(
  const mx::array& inputs,
  const mx::array& targets,
  const s::optional<mx::array>& weights,
  bool with_logits,
  Reduction reduction)
{
  check_same_shape("Inputs", inputs, targets);
  mx::array loss = inputs;
  if (with_logits) {
    loss = mx::logaddexp(scalar_like(0, inputs), inputs) - inputs * targets;
  } else {
    mx::array lo = scalar_like(-100, inputs);
    mx::array log_in = mx::maximum(mx::log(inputs), lo);
    mx::array log_inv = mx::maximum(mx::log(scalar_like(1, inputs) - inputs), lo);
    loss = -(targets * log_in + (scalar_like(1, targets) - targets) * log_inv);
  }
  apply_weights(loss, weights);
  return reduce(loss, reduction);
}

mx::array l1_loss(
  const mx::array& predictions,
  const mx::array& targets,
  Reduction reduction)
{
  check_same_shape("Predictions", predictions, targets);
  return reduce(mx::abs(predictions - targets), reduction);
}

mx::array mse_loss(
  const mx::array& predictions,
  const mx::array& targets,
  Reduction reduction)
{
  check_same_shape("Predictions", predictions, targets);
  return reduce(mx::square(predictions - targets), reduction);
}

mx::array nll_loss(
  const mx::array& inputs,
  const mx::array& targets,
  int axis,
  Reduction reduction)
{
  mx::array loss = -mx::squeeze(
    mx::take_along_axis(inputs, mx::expand_dims(targets, axis), axis),
    axis
  );
  return reduce(loss, reduction);
}

mx::array gaussian_nll_loss(
  const mx::array& inputs,
  const mx::array& targets,
  const mx::array& vars,
  bool full,
  double eps,
  Reduction reduction)
{
  check_same_shape("Inputs", inputs, targets);
  if (inputs.shape() != vars.shape())
    throw s::invalid_argument(
      "Inputs shape " + shape_str(inputs) + " does not match vars shape " +
      shape_str(vars) + "."
    );
  mx::array v = mx::maximum(vars, scalar_like(eps, vars));
  mx::array loss = scalar_like(0.5, v) *
                   (mx::log(v) + mx::square(targets - inputs) / v);
  if (full) loss = loss + scalar_like(0.5 * std::log(2 * s::numbers::pi), loss);
  return reduce(loss, reduction);
}

mx::array kl_div_loss(
  const mx::array& inputs,
  const mx::array& targets,
  int axis,
  Reduction reduction)
{
  return reduce(mx::sum(mx::exp(targets) * (targets - inputs), axis), reduction);
}

mx::array smooth_l1_loss(
  const mx::array& predictions,
  const mx::array& targets,
  double beta,
  Reduction reduction)
{
  check_same_shape("Predictions", predictions, targets);
  mx::array diff = mx::abs(predictions - targets);
  mx::array b = scalar_like(beta, diff);
  mx::array loss = mx::where(
    mx::less(diff, b),
    scalar_like(0.5, diff) * mx::square(diff) / b,
    diff - scalar_like(0.5 * beta, diff)
  );
  return reduce(loss, reduction);
}

mx::array triplet_loss(
  const mx::array& anchors,
  const mx::array& positives,
  const mx::array& negatives,
  int axis,
  int p,
  double margin,
  double eps,
  Reduction reduction)
{
  auto dist = [&](const mx::array& other) {
    mx::array d = mx::abs(anchors - other);
    mx::array sum = mx::sum(mx::power(d, scalar_like(p, d)), axis) + scalar_like(eps, d);
    return mx::power(sum, scalar_like(1.0 / p, sum));
  };
  mx::array pos = dist(positives);
  mx::array neg = dist(negatives);
  mx::array loss = mx::maximum(pos - neg + scalar_like(margin, pos), scalar_like(0, pos));
  return reduce(loss, reduction);
}

mx::array hinge_loss(
  const mx::array& inputs,
  const mx::array& targets,
  Reduction reduction)
{
  mx::array loss = mx::maximum(
    scalar_like(1, inputs) - inputs * targets,
    scalar_like(0, inputs)
  );
  return reduce(loss, reduction);
}

mx::array huber_loss(
  const mx::array& inputs,
  const mx::array& targets,
  double delta,
  Reduction reduction)
{
  mx::array abs_errors = mx::abs(inputs - targets);
  mx::array quadratic = mx::minimum(abs_errors, scalar_like(delta, abs_errors));
  mx::array linear = abs_errors - quadratic;
  mx::array loss = scalar_like(0.5, quadratic) * mx::square(quadratic) +
                   scalar_like(delta, linear) * linear;
  return reduce(loss, reduction);
}

mx::array log_cosh_loss(
  const mx::array& inputs,
  const mx::array& targets,
  Reduction reduction)
{
  mx::array errors = inputs - targets;
  mx::array loss = mx::logaddexp(errors, -errors) - scalar_like(s::numbers::ln2, errors);
  return reduce(loss, reduction);
}

mx::array cosine_similarity_loss(
  const mx::array& x1,
  const mx::array& x2,
  int axis,
  double eps,
  Reduction reduction)
{
  mx::array x1_norm = mx::linalg::norm(x1, axis);
  mx::array x2_norm = mx::linalg::norm(x2, axis);
  mx::array loss = mx::sum(x1 * x2, axis) /
                   mx::maximum(x1_norm * x2_norm, scalar_like(eps, x1_norm));
  return reduce(loss, reduction);
}

mx::array margin_ranking_loss(
  const mx::array& inputs1,
  const mx::array& inputs2,
  const mx::array& targets,
  double margin,
  Reduction reduction)
{
  if (inputs1.shape() != inputs2.shape() or inputs1.shape() != targets.shape())
    throw s::invalid_argument(
      "The shapes of the arguments do not match. The provided shapes are "
      "inputs1.shape=" + shape_str(inputs1) + ", inputs2.shape=" +
      shape_str(inputs2) + ", and targets.shape=" + shape_str(targets) + "."
    );
  mx::array differences = inputs1 - inputs2;
  mx::array loss = mx::maximum(
    scalar_like(0, differences),
    -targets * differences + scalar_like(margin, differences)
  );
  return reduce(loss, reduction);
}

} // namespace rlbg::nn
