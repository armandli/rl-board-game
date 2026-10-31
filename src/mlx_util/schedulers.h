#ifndef RLBG_MLX_UTIL_SCHEDULERS_H
#define RLBG_MLX_UTIL_SCHEDULERS_H

#include <functional>
#include <vector>

#include <mlx/mlx.h>

namespace rlbg::optim {

namespace s = std;
namespace mx = mlx::core;

// maps the optimizer's step counter (a uint64 scalar) to a hyperparameter
using Schedule = s::function<mx::array(const mx::array& step)>;

// mlx.optimizers.schedulers
Schedule exponential_decay(double init, double decay_rate);
Schedule step_decay(double init, double decay_rate, int step_size);
Schedule cosine_decay(double init, int decay_steps, double end = 0.0);
// schedules[i + 1] starts at boundaries[i], seeing steps counted from there
Schedule join_schedules(s::vector<Schedule> schedules, s::vector<int> boundaries);
Schedule linear_schedule(double init, double end, int steps);

} // namespace rlbg::optim

#endif//RLBG_MLX_UTIL_SCHEDULERS_H
