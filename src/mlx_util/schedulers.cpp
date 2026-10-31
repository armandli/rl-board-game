#include <mlx_util/schedulers.h>

#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace rlbg::optim {

namespace {

mx::array f32(double v) { return mx::array((float)v); }

// integer constants take the step's dtype, as Python ints do
mx::array step_scalar(int v, const mx::array& step) { return mx::array(v, step.dtype()); }

} // namespace

Schedule exponential_decay(double init, double decay_rate) {
  return [=](const mx::array& step) {
    return f32(init) * mx::power(f32(decay_rate), step);
  };
}

Schedule step_decay(double init, double decay_rate, int step_size) {
  if (step_size < 1)
    throw s::invalid_argument(
      "step_size must be greater than 0, but got " + s::to_string(step_size) + "."
    );
  return [=](const mx::array& step) {
    return f32(init) *
           mx::power(f32(decay_rate), mx::floor_divide(step, step_scalar(step_size, step)));
  };
}

Schedule cosine_decay(double init, int decay_steps, double end) {
  if (decay_steps < 1)
    throw s::invalid_argument(
      "decay_steps must be greater than 0, but got " + s::to_string(decay_steps) + "."
    );
  return [=](const mx::array& step) {
    mx::array clamped = mx::minimum(step, step_scalar(decay_steps, step));
    mx::array decay =
      f32(0.5) * (f32(1.0) + mx::cos(f32(s::numbers::pi / decay_steps) * clamped));
    return f32(end) + decay * f32(init - end);
  };
}

Schedule join_schedules(s::vector<Schedule> schedules, s::vector<int> boundaries) {
  if (schedules.empty())
    throw s::invalid_argument("Must provide at least 1 schedule to join.");
  if (schedules.size() != boundaries.size() + 1)
    throw s::invalid_argument(
      "Received " + s::to_string(boundaries.size()) + " boundaries but expected " +
      s::to_string(schedules.size() - 1) + "."
    );
  return [schedules = s::move(schedules), boundaries = s::move(boundaries)](
           const mx::array& step) {
    mx::array output = schedules[0](step);
    for (size_t i = 0; i < boundaries.size(); ++i) {
      mx::array boundary = step_scalar(boundaries[i], step);
      output = mx::where(mx::less(step, boundary), output, schedules[i + 1](step - boundary));
    }
    return output;
  };
}

Schedule linear_schedule(double init, double end, int steps) {
  if (steps < 1)
    throw s::invalid_argument(
      "steps must be greater than 0, but got " + s::to_string(steps) + "."
    );
  return [=](const mx::array& step) {
    mx::array clamped = mx::minimum(step, step_scalar(steps, step));
    return clamped * f32((end - init) / steps) + f32(init);
  };
}

} // namespace rlbg::optim
