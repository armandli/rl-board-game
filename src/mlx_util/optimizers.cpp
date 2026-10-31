#include <mlx_util/optimizers.h>

#include <cmath>

#include <algorithm>
#include <stdexcept>

#include <mlx_util/io.h>

namespace rlbg::optim {

namespace {

mx::array like(double v, const mx::array& x) { return nn::scalar_like(v, x); }

mx::array f32(double v) { return mx::array((float)v); }

mx::array rms(const mx::array& x) { return mx::sqrt(mx::mean(mx::square(x))); }

void check_betas(const char* name, s::pair<double, double> betas) {
  for (int i = 0; double beta : {betas.first, betas.second}) {
    ++i;
    if (not (0.0 <= beta and beta < 1.0))
      throw s::invalid_argument(
        s::string(name) + " beta" + s::to_string(i) +
        " should be in [0, 1), " + s::to_string(beta) + " was provided instead"
      );
  }
}

} // namespace

Optimizer::Optimizer() {
  mScalars.insert_or_assign("step", mx::array(0, mx::uint64));
}

void Optimizer::maybe_schedule(const s::string& name, const Hyper& value) {
  if (value.schedule) {
    mSchedulers.insert_or_assign(name, value.schedule);
    mScalars.insert_or_assign(name, value.schedule(step()));
  } else {
    mScalars.insert_or_assign(name, f32(value.value));
  }
}

void Optimizer::init(const Params& parameters) {
  for (const auto& [key, p] : parameters) {
    Slots& slots = mSlots[key];
    if (slots.empty()) init_single(p, slots);
  }
  mInitialized = true;
}

Params Optimizer::apply_gradients(const Params& gradients, const Params& parameters) {
  if (not mInitialized) init(gradients);
  for (const auto& [name, schedule] : mSchedulers)
    mScalars.insert_or_assign(name, schedule(step()));
  mScalars.insert_or_assign("step", step() + mx::array(1, step().dtype()));

  Params out;
  for (const auto& [key, g] : gradients) {
    auto p = parameters.find(key);
    if (p == parameters.end())
      throw s::invalid_argument("[Optimizer] no parameter for gradient " + key);
    out.insert_or_assign(key, apply_single(g, p->second, mSlots[key]));
  }
  return out;
}

Params Optimizer::state() const {
  Params out = mScalars;
  for (const auto& [key, slots] : mSlots)
    for (const auto& [name, v] : slots) out.insert_or_assign(key + "." + name, v);
  return out;
}

void Optimizer::set_state(const Params& state) {
  // slot names never contain '.', so the last dot separates the parameter
  // key from the slot; dotless keys are top level scalars
  Params scalars;
  s::map<s::string, Slots> slots;
  for (const auto& [key, v] : state) {
    size_t dot = key.rfind('.');
    if (dot == s::string::npos) scalars.insert_or_assign(key, v);
    else slots[key.substr(0, dot)].insert_or_assign(key.substr(dot + 1), v);
  }
  // keep hyperparameters the state does not mention instead of failing on
  // the next step the way Python would
  for (const auto& [key, v] : mScalars)
    if (not scalars.contains(key)) scalars.insert_or_assign(key, v);
  mScalars = s::move(scalars);
  mSlots = s::move(slots);
  mInitialized = false;
}

void Optimizer::save_state(const s::string& file) const { nn::save_arrays(file, state()); }

void Optimizer::load_state(const s::string& file) { set_state(nn::load_arrays(file)); }

SGD::SGD(Hyper learning_rate, double momentum, double weight_decay, double dampening, bool nesterov)
  : momentum(momentum), weight_decay(weight_decay), dampening(dampening), nesterov(nesterov)
{
  if (nesterov and (momentum <= 0 or dampening != 0))
    throw s::invalid_argument("Nesterov momentum requires a momentum and zero dampening.");
  maybe_schedule("learning_rate", learning_rate);
}

void SGD::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("v", mx::zeros_like(parameter));
}

mx::array SGD::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  mx::array g = gradient;
  if (weight_decay != 0) g = g + like(weight_decay, g) * parameter;
  if (momentum <= 0) return parameter - lr_like(g) * g;

  mx::array v = like(momentum, g) * state.at("v");
  v = dampening > 0 ? v + like(1 - dampening, g) * g : v + g;
  mx::array update = nesterov ? g + like(momentum, g) * v : v;
  state.insert_or_assign("v", v);
  return parameter - lr_like(g) * update;
}

RMSprop::RMSprop(Hyper learning_rate, double alpha, double eps) : alpha(alpha), eps(eps) {
  maybe_schedule("learning_rate", learning_rate);
  if (alpha < 0.0)
    throw s::invalid_argument(
      "RMSprop alpha should be >=0, " + s::to_string(alpha) + " was provided instead"
    );
  if (eps <= 0.0)
    throw s::invalid_argument(
      "RMSprop epsilon should be >0, " + s::to_string(eps) + " was provided instead"
    );
}

void RMSprop::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("v", mx::zeros_like(parameter));
}

mx::array RMSprop::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  mx::array v = like(alpha, gradient) * state.at("v") +
                like(1 - alpha, gradient) * mx::square(gradient);
  state.insert_or_assign("v", v);
  return parameter - lr_like(gradient) * gradient / (mx::sqrt(v) + like(eps, gradient));
}

Adagrad::Adagrad(Hyper learning_rate, double eps) : eps(eps) {
  maybe_schedule("learning_rate", learning_rate);
  if (eps <= 0.0)
    throw s::invalid_argument(
      "Adagrad epsilon should be >0, " + s::to_string(eps) + " was provided instead"
    );
}

void Adagrad::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("v", mx::zeros_like(parameter));
}

mx::array Adagrad::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  mx::array v = state.at("v") + mx::square(gradient);
  state.insert_or_assign("v", v);
  return parameter - lr_like(gradient) * gradient / (mx::sqrt(v) + like(eps, gradient));
}

AdaDelta::AdaDelta(Hyper learning_rate, double rho, double eps) : rho(rho), eps(eps) {
  maybe_schedule("learning_rate", learning_rate);
  if (rho < 0.0)
    throw s::invalid_argument(
      "AdaDelta rho should be >=0, " + s::to_string(rho) + " was provided instead"
    );
  if (eps <= 0.0)
    throw s::invalid_argument(
      "AdaDelta epsilon should be >0, " + s::to_string(eps) + " was provided instead"
    );
}

void AdaDelta::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("v", mx::zeros_like(parameter));
  state.insert_or_assign("u", mx::zeros_like(parameter));
}

mx::array AdaDelta::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  const mx::array& g = gradient;
  mx::array e = like(eps, g);
  mx::array v = like(rho, g) * state.at("v") + like(1 - rho, g) * mx::square(g);
  mx::array d = mx::sqrt(state.at("u") + e) / mx::sqrt(v + e) * g;
  mx::array u = like(rho, g) * state.at("u") + like(1 - rho, g) * mx::square(d);
  state.insert_or_assign("v", v);
  state.insert_or_assign("u", u);
  return parameter - lr_like(g) * d;
}

Adam::Adam(Hyper learning_rate, s::pair<double, double> betas, double eps, bool bias_correction)
  : betas(betas), eps(eps), bias_correction(bias_correction)
{
  check_betas("Adam", betas);
  if (not (0.0 <= eps))
    throw s::invalid_argument(
      "Adam epsilon should be >=0, " + s::to_string(eps) + " was provided instead"
    );
  maybe_schedule("learning_rate", learning_rate);
}

void Adam::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("m", mx::zeros_like(parameter));
  state.insert_or_assign("v", mx::zeros_like(parameter));
}

mx::array Adam::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  const mx::array& g = gradient;
  mx::array lr = lr_like(g);
  auto [b1, b2] = betas;
  mx::array m = like(b1, g) * state.at("m") + like(1 - b1, g) * g;
  mx::array v = like(b2, g) * state.at("v") + like(1 - b2, g) * mx::square(g);
  state.insert_or_assign("m", m);
  state.insert_or_assign("v", v);
  if (bias_correction) {
    mx::array c1 = mx::astype(lr / (f32(1) - mx::power(f32(b1), step())), g.dtype());
    mx::array c2 = mx::astype(mx::rsqrt(f32(1) - mx::power(f32(b2), step())), g.dtype());
    return parameter - c1 * m / (mx::sqrt(v) * c2 + like(eps, g));
  }
  return parameter - lr * m / (mx::sqrt(v) + like(eps, g));
}

AdamW::AdamW(
  Hyper learning_rate,
  s::pair<double, double> betas,
  double eps,
  double weight_decay,
  bool bias_correction)
  : Adam(s::move(learning_rate), betas, eps, bias_correction), weight_decay(weight_decay)
{}

mx::array AdamW::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  mx::array lr = lr_like(gradient);
  mx::array decayed =
    parameter * (like(1, lr) - lr * like(weight_decay, lr));
  return Adam::apply_single(gradient, decayed, state);
}

Adamax::Adamax(Hyper learning_rate, s::pair<double, double> betas, double eps)
  : Adam(s::move(learning_rate), betas, eps)
{}

mx::array Adamax::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  const mx::array& g = gradient;
  auto [b1, b2] = betas;
  mx::array m = like(b1, g) * state.at("m") + like(1 - b1, g) * g;
  mx::array v = mx::maximum(like(b2, g) * state.at("v"), mx::abs(g));
  state.insert_or_assign("m", m);
  state.insert_or_assign("v", v);
  return parameter - lr_like(g) * m / (v + like(eps, g));
}

Lion::Lion(Hyper learning_rate, s::pair<double, double> betas, double weight_decay)
  : betas(betas), weight_decay(weight_decay)
{
  check_betas("Lion", betas);
  maybe_schedule("learning_rate", learning_rate);
}

void Lion::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("m", mx::zeros_like(parameter));
}

mx::array Lion::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  const mx::array& g = gradient;
  mx::array lr = lr_like(g);
  auto [b1, b2] = betas;
  const mx::array m = state.at("m");
  mx::array c = like(b1, g) * m + like(1 - b1, g) * g;
  state.insert_or_assign("m", like(b2, g) * m + like(1 - b2, g) * g);
  mx::array p = parameter;
  if (weight_decay > 0) p = (like(1, lr) - lr * like(weight_decay, lr)) * p;
  return p - lr * mx::sign(c);
}

Adafactor::Adafactor(
  s::optional<Hyper> learning_rate,
  s::pair<double, double> eps,
  double clip_threshold,
  double decay_rate,
  s::optional<double> beta_1,
  double weight_decay,
  bool scale_parameter,
  bool relative_step,
  bool warmup_init)
  : eps(eps),
    clip_threshold(clip_threshold),
    decay_rate(decay_rate),
    beta_1(beta_1),
    weight_decay(weight_decay),
    scale_parameter(scale_parameter),
    relative_step(relative_step),
    warmup_init(warmup_init)
{
  if (learning_rate) maybe_schedule("learning_rate", *learning_rate);
}

void Adafactor::init_single(const mx::array& parameter, Slots& state) {
  if (parameter.ndim() >= 2) {
    const mx::Shape& shape = parameter.shape();
    mx::Shape row(shape.begin(), shape.end() - 1);
    mx::Shape col(shape.begin(), shape.end() - 2);
    col.push_back(shape.back());
    state.insert_or_assign("exp_avg_sq_row", mx::zeros(row, parameter.dtype()));
    state.insert_or_assign("exp_avg_sq_col", mx::zeros(col, parameter.dtype()));
  } else {
    state.insert_or_assign("exp_avg_sq", mx::zeros_like(parameter));
  }
  if (beta_1) state.insert_or_assign("exp_avg", mx::zeros_like(parameter));
}

mx::array Adafactor::compute_learning_rate(const mx::array& step, const mx::array& parameter_rms) const {
  mx::array relative_step_size = f32(0);
  if (relative_step) {
    mx::array min_step = warmup_init ? f32(1e-6) * step : f32(1e-2);
    relative_step_size = mx::minimum(min_step, mx::rsqrt(step));
  } else {
    if (not mScalars.contains("learning_rate"))
      throw s::invalid_argument("Adafactor needs a learning_rate when relative_step is false");
    relative_step_size = learning_rate();
  }
  relative_step_size = mx::astype(relative_step_size, parameter_rms.dtype());
  mx::array parameter_scale = scale_parameter
    ? mx::maximum(like(eps.second, parameter_rms), parameter_rms)
    : like(1, parameter_rms);
  return parameter_scale * relative_step_size;
}

mx::array Adafactor::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  const mx::array& g = gradient;
  bool factored = g.ndim() >= 2;
  mx::array st = step();
  mx::array parameter_rms = rms(parameter);
  mx::array lr = compute_learning_rate(st, parameter_rms);
  mx::array beta_2 = like(1, parameter_rms) - mx::astype(
    mx::power(mx::astype(st, mx::float32), f32(decay_rate)),
    parameter_rms.dtype()
  );
  mx::array one_minus_beta_2 = like(1, beta_2) - beta_2;
  mx::array update = mx::square(g) + like(eps.first, g);

  if (factored) {
    mx::array row = beta_2 * state.at("exp_avg_sq_row") + one_minus_beta_2 * mx::mean(update, -1);
    mx::array col = beta_2 * state.at("exp_avg_sq_col") + one_minus_beta_2 * mx::mean(update, -2);
    state.insert_or_assign("exp_avg_sq_row", row);
    state.insert_or_assign("exp_avg_sq_col", col);
    mx::array r_factor = mx::rsqrt(row / mx::mean(row, -1, true));
    mx::array c_factor = mx::rsqrt(col);
    update = mx::expand_dims(r_factor, -1) * mx::expand_dims(c_factor, -2) * g;
  } else {
    mx::array avg = beta_2 * state.at("exp_avg_sq") + one_minus_beta_2 * update;
    state.insert_or_assign("exp_avg_sq", avg);
    update = mx::rsqrt(avg) * g;
  }
  update = update / mx::maximum(like(1, update), rms(update) / like(clip_threshold, update));
  update = lr * update;
  if (beta_1) {
    mx::array avg = like(*beta_1, update) * state.at("exp_avg") + like(1 - *beta_1, update) * update;
    state.insert_or_assign("exp_avg", avg);
    update = avg;
  }
  mx::array p = parameter;
  if (weight_decay != 0) p = p + p * (like(-weight_decay, lr) * lr);
  return p - update;
}

Muon::Muon(Hyper learning_rate, double momentum, double weight_decay, bool nesterov, int ns_steps)
  : momentum(momentum), weight_decay(weight_decay), nesterov(nesterov), ns_steps(ns_steps)
{
  maybe_schedule("learning_rate", learning_rate);
}

void Muon::init_single(const mx::array& parameter, Slots& state) {
  state.insert_or_assign("v", mx::zeros_like(parameter));
}

namespace {

// quintic Newton-Schulz iteration approximating the orthogonal factor of X;
// the coefficients are the ones Muon's authors tuned for fast convergence
mx::array zeropower_via_newtonschulz5(const mx::array& x, int steps) {
  if (x.ndim() != 2)
    throw s::invalid_argument("Expected a 2D array for Newton-Schulz iteration");
  const float a = 3.4445f;
  const float b = -4.7750f;
  const float c = 2.0315f;
  bool transpose_needed = x.shape(-2) > x.shape(-1);
  mx::array X = transpose_needed ? mx::transpose(x) : x;
  X = X / (mx::linalg::norm(X, s::nullopt, true) + like(1e-7, X));
  for (int i = 0; i < steps; ++i) {
    mx::array A = mx::matmul(X, mx::transpose(X));
    mx::array B = mx::addmm(like(b, A) * A, A, A, c, 1.0f);
    X = mx::addmm(like(a, X) * X, B, X, 1.0f, 1.0f);
  }
  return transpose_needed ? mx::transpose(X) : X;
}

} // namespace

mx::array Muon::apply_single(const mx::array& gradient, const mx::array& parameter, Slots& state) {
  mx::array g = gradient;
  if (weight_decay != 0) g = g + like(weight_decay, g) * parameter;
  mx::array v = like(momentum, g) * state.at("v") + like(1 - momentum, g) * g;
  state.insert_or_assign("v", v);
  mx::array update = nesterov ? g * like(1 - momentum, g) + v * like(momentum, g) : v;
  mx::array lr = lr_like(g);
  if (update.ndim() >= 2) {
    mx::Shape original = update.shape();
    bool reshape_needed = update.ndim() > 2;
    if (reshape_needed) update = mx::reshape(update, {update.shape(0), -1});
    update = zeropower_via_newtonschulz5(update, ns_steps);
    if (reshape_needed) update = mx::reshape(update, original);
    // Python takes the ratio from the restored shape's last two axes
    double ratio = (double)update.shape(-2) / update.shape(-1);
    lr = lr * like(std::sqrt(s::max(1.0, ratio)), lr);
  }
  return parameter - lr * update;
}

MultiOptimizer::MultiOptimizer(s::vector<s::shared_ptr<Optimizer>> optimizers, s::vector<Filter> filters)
  : mOptimizers(s::move(optimizers)), mFilters(s::move(filters))
{
  if (mFilters.size() + 1 != mOptimizers.size())
    throw s::invalid_argument(
      "Given " + s::to_string(mFilters.size()) + " filters but " +
      s::to_string(mOptimizers.size() - 1) + " needed."
    );
  mFilters.push_back([](const s::string&, const mx::array&) { return true; });
  mScalars.clear();
}

s::vector<Params> MultiOptimizer::split(const Params& gradients) const {
  if (mOptimizers.size() == 1) return {gradients};
  s::vector<Params> parts(mOptimizers.size());
  for (const auto& [key, g] : gradients)
    for (size_t i = 0; i < mFilters.size(); ++i)
      if (mFilters[i](key, g)) {
        parts[i].insert_or_assign(key, g);
        break;
      }
  return parts;
}

void MultiOptimizer::init(const Params& parameters) {
  s::vector<Params> parts = split(parameters);
  for (size_t i = 0; i < mOptimizers.size(); ++i) mOptimizers[i]->init(parts[i]);
}

Params MultiOptimizer::apply_gradients(const Params& gradients, const Params& parameters) {
  s::vector<Params> parts = split(gradients);
  Params out;
  for (size_t i = 0; i < mOptimizers.size(); ++i)
    for (auto& [k, v] : mOptimizers[i]->apply_gradients(parts[i], parameters))
      out.insert_or_assign(k, v);
  return out;
}

Params MultiOptimizer::state() const {
  Params out;
  for (size_t i = 0; i < mOptimizers.size(); ++i)
    for (const auto& [k, v] : mOptimizers[i]->state())
      out.insert_or_assign("states." + s::to_string(i) + "." + k, v);
  return out;
}

void MultiOptimizer::set_state(const Params& state) {
  const s::string prefix = "states.";
  s::vector<Params> parts(mOptimizers.size());
  s::vector<bool> seen(mOptimizers.size(), false);
  for (const auto& [key, v] : state) {
    size_t dot = key.find('.', prefix.size());
    if (not key.starts_with(prefix) or dot == s::string::npos)
      throw s::invalid_argument("Invalid state provided");
    size_t i = s::stoul(key.substr(prefix.size(), dot - prefix.size()));
    if (i >= mOptimizers.size()) throw s::invalid_argument("Invalid state provided");
    parts[i].insert_or_assign(key.substr(dot + 1), v);
    seen[i] = true;
  }
  if (s::ranges::find(seen, false) != seen.end())
    throw s::invalid_argument("Invalid state provided");
  for (size_t i = 0; i < mOptimizers.size(); ++i) mOptimizers[i]->set_state(parts[i]);
}

void MultiOptimizer::set_learning_rate(double learning_rate) {
  for (auto& o : mOptimizers) o->set_learning_rate(learning_rate);
}

s::pair<Params, mx::array> clip_grad_norm(const Params& gradients, double max_norm) {
  if (max_norm < 0)
    throw s::invalid_argument(
      "max_norm should be >=0, " + s::to_string(max_norm) + " was provided instead"
    );
  mx::array norm_squared = f32(0);
  for (const auto& [k, g] : gradients) {
    mx::array gg = g.dtype() == mx::float16 ? mx::astype(g, mx::float32) : g;
    norm_squared = norm_squared + mx::sum(mx::square(gg));
  }
  mx::array total_norm = mx::sqrt(norm_squared);
  mx::array normalizer = mx::minimum(f32(max_norm) / (total_norm + f32(1e-6)), f32(1.0));
  Params clipped;
  for (const auto& [k, g] : gradients) {
    mx::array scaled = g * normalizer;
    if (mx::issubdtype(g.dtype(), mx::floating)) scaled = mx::astype(scaled, g.dtype());
    clipped.insert_or_assign(k, scaled);
  }
  return {clipped, total_norm};
}

} // namespace rlbg::optim
