#include <libtorch/learner.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>

#include <ATen/autocast_mode.h>

namespace rlbg::libtorch {

namespace sf = std::filesystem;

namespace {

// quintic Newton-Schulz iteration approximating the orthogonal factor of x,
// with the coefficients Muon's authors tuned for fast convergence
t::Tensor zeropower_via_newtonschulz5(const t::Tensor& x, int steps){
  const double a = 3.4445, b = -4.7750, c = 2.0315;
  bool transpose = x.size(0) > x.size(1);
  t::Tensor X = (transpose ? x.t() : x).to(t::kBFloat16);
  X = X / (X.norm() + 1e-7);
  for (int i = 0; i < steps; ++i){
    t::Tensor A = X.matmul(X.t());
    t::Tensor B = b * A + c * A.matmul(A);
    X = a * X + B.matmul(X);
  }
  return (transpose ? X.t() : X).to(x.scalar_type());
}

// sum over the batch of masked losses, divided by the batch size
t::Tensor masked_mean(const t::Tensor& loss, const t::Tensor& mask, int batch){
  return (loss * mask).sum() / (double)batch;
}

// logits (..., C), target classes (...) -> per-element cross entropy
t::Tensor class_ce(const t::Tensor& logits, const t::Tensor& target){
  t::Tensor logp = t::log_softmax(logits.to(t::kFloat), -1);
  return -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1);
}

// logits (B, C), target distributions (B, C) -> (B,)
t::Tensor soft_ce(const t::Tensor& logits, const t::Tensor& target){
  return -(target * t::log_softmax(logits.to(t::kFloat), -1)).sum(-1);
}

// write then rename, so a crash never leaves a half written file
template <typename F>
void write_atomically(const sf::path& path, F write){
  sf::path tmp = path;
  tmp += ".tmp";
  write(tmp.string());
  sf::rename(tmp, path);
}

} // namespace

Muon::Muon(s::vector<t::Tensor> params, double momentum, double weight_decay, int ns_steps)
: params(s::move(params)), momentum(momentum), weight_decay(weight_decay), ns_steps(ns_steps)
{
  for (const t::Tensor& p : this->params) momentum_buffers.push_back(t::zeros_like(p));
}

void Muon::step(double lr){
  t::NoGradGuard no_grad;
  for (size_t i = 0; i < params.size(); ++i){
    t::Tensor& p = params[i];
    if (not p.grad().defined()) continue;
    t::Tensor g = p.grad();
    if (weight_decay != 0.) g = g + weight_decay * p;
    t::Tensor& v = momentum_buffers[i];
    v.mul_(momentum).add_(g, 1. - momentum);
    t::Tensor update = g * (1. - momentum) + v * momentum;
    update = zeropower_via_newtonschulz5(update, ns_steps);
    double ratio = (double)p.size(0) / (double)p.size(1);
    p.add_(update, -lr * s::sqrt(s::max(1., ratio)));
  }
}

void Muon::save(t::serialize::OutputArchive& archive) const {
  for (size_t i = 0; i < momentum_buffers.size(); ++i)
    archive.write("v" + s::to_string(i), momentum_buffers[i]);
}

void Muon::load(t::serialize::InputArchive& archive){
  for (size_t i = 0; i < momentum_buffers.size(); ++i){
    t::Tensor v;
    archive.read("v" + s::to_string(i), v);
    momentum_buffers[i].copy_(v);
  }
}

Learner::Learner(const TransformerMuZeroConfig& model, const LearnerConfig& config, t::Device device)
: mModel(model), mConfig(config), mDevice(device), mStream(device)
{
  if (config.optimizer != "muon" and config.optimizer != "adamw")
    throw s::invalid_argument("unknown optimizer " + config.optimizer);
  DeviceStream::Scope scope(mStream);
  mModel->to(device);
  mModel->train();
  // Muon takes the blocks' hidden matrices; AdamW decays the other matrices
  // but not norms, biases, embeddings
  s::vector<t::Tensor> muon, decay, no_decay;
  for (const auto& item : mModel->named_parameters(true)){
    const s::string& name = item.key();
    bool matrix = item.value().dim() == 2 and name.find("embedding") == s::string::npos;
    bool hidden = matrix and name.find("_blocks.") != s::string::npos;
    if (hidden and config.optimizer == "muon") muon.push_back(item.value());
    else if (matrix)                           decay.push_back(item.value());
    else                                       no_decay.push_back(item.value());
  }
  auto adamw = [&](double wd){
    return s::make_unique<t::optim::AdamWOptions>(
      t::optim::AdamWOptions(config.learning_rate).weight_decay(wd).betas({0.9, 0.95}).eps(1e-8));
  };
  s::vector<t::optim::OptimizerParamGroup> groups;
  groups.emplace_back(decay, adamw(config.weight_decay));
  groups.emplace_back(no_decay, adamw(0.));
  mAdam = s::make_unique<t::optim::AdamW>(groups);
  if (not muon.empty()) mMuon = s::make_unique<Muon>(muon, 0.95, config.weight_decay);
}

double Learner::schedule(int64_t step) const {
  const LearnerConfig& c = mConfig;
  int warmup = s::max(1, c.warmup_steps);
  if (step < warmup) return 0.01 + 0.99 * (double)step / warmup;
  double progress = s::min(1., (double)(step - warmup) / s::max(1, c.total_steps - warmup));
  double cosine = 0.5 * (1. + s::cos(s::numbers::pi * progress));
  return c.final_lr_fraction + (1. - c.final_lr_fraction) * cosine;
}

LearnerStats Learner::train_step(const TrainingBatch& batch){
  const LearnerConfig& c = mConfig;
  const int B = batch.B, K = batch.K, T = batch.T, F = batch.F, A = batch.A;
  DeviceStream::Scope scope(mStream);
  TransformerMuZero& model = mModel;

  auto upload = [&](const auto& v, s::vector<int64_t> shape, t::ScalarType type){
    using V = typename s::decay_t<decltype(v)>::value_type;
    return t::from_blob(const_cast<V*>(v.data()), shape, type).to(mDevice);
  };
  t::Tensor obs = upload(batch.observations, {B, T, F}, t::kUInt8);
  t::Tensor target_obs = upload(batch.target_observations, {B * K, T, F}, t::kUInt8);
  t::Tensor actions = upload(batch.actions, {B, K}, t::kInt).to(t::kLong);
  t::Tensor policies = upload(batch.policies, {B, K + 1, A}, t::kFloat);
  t::Tensor policy_mask = upload(batch.policy_mask, {B, K + 1}, t::kFloat);
  t::Tensor values = upload(batch.values, {B, K + 1, NUM_OUTCOME_CLASSES}, t::kFloat);
  t::Tensor value_mask = upload(batch.value_mask, {B, K + 1}, t::kFloat);
  t::Tensor rewards = upload(batch.rewards, {B, K}, t::kInt).to(t::kLong);
  t::Tensor reward_mask = upload(batch.reward_mask, {B, K}, t::kFloat);
  t::Tensor tokens = upload(batch.tokens, {B, K + 1, T}, t::kInt).to(t::kLong);
  t::Tensor ownership = upload(batch.ownership, {B, K + 1, T}, t::kInt).to(t::kLong);
  t::Tensor state_mask = upload(batch.state_mask, {B, K + 1}, t::kFloat);

  double mult = schedule(mStep);
  double lr = c.learning_rate * mult;
  for (auto& group : mAdam->param_groups())
    static_cast<t::optim::AdamWOptions&>(group.options()).lr(lr);

  bool autocast = c.bf16 and mDevice.is_cuda();
  if (autocast){
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
  }

  t::Tensor zero = t::zeros({}, t::TensorOptions().device(mDevice));
  t::Tensor lp = zero, lv = zero, lr_ = zero, lc = zero, lg = zero, lo = zero;

  // consistency targets: projections of the real future states, no gradient
  t::Tensor target_proj;
  if (K > 0){
    t::NoGradGuard no_grad;
    target_proj = model->project(model->represent(target_obs)).to(t::kFloat).view({B, K, -1});
    target_proj = t::nn::functional::normalize(target_proj, t::nn::functional::NormalizeFuncOptions().dim(-1));
  }

  t::Tensor latent = model->represent(obs);
  for (int k = 0; k <= K; ++k){
    double w = k == 0 ? 1. : 1. / K;
    Prediction pred = model->predict(latent);
    lp = lp + w * masked_mean(soft_ce(pred.policy_logits, policies.select(1, k)), policy_mask.select(1, k), B);
    lv = lv + w * masked_mean(soft_ce(pred.value_logits, values.select(1, k)), value_mask.select(1, k), B);
    t::Tensor own = class_ce(pred.ownership_logits, ownership.select(1, k)).mean(-1);
    lo = lo + w * masked_mean(own, value_mask.select(1, k), B);
    t::Tensor recon = class_ce(model->reconstruct(latent), tokens.select(1, k)).mean(-1);
    lg = lg + w * masked_mean(recon, state_mask.select(1, k), B);

    if (k > 0){
      t::Tensor p = model->predict_projection(model->project(latent)).to(t::kFloat).view({B, -1});
      p = t::nn::functional::normalize(p, t::nn::functional::NormalizeFuncOptions().dim(-1));
      t::Tensor cos = (p * target_proj.select(1, k - 1)).sum(-1);
      lc = lc + masked_mean(-cos, state_mask.select(1, k), B) / (double)K;
    }
    if (k < K){
      double g = c.dynamics_gradient_scale;
      t::Tensor scaled = latent * g + latent.detach() * (1. - g);
      DynamicsOutput dyn = model->dynamics(scaled, actions.select(1, k));
      lr_ = lr_ + masked_mean(class_ce(dyn.reward_logits, rewards.select(1, k)), reward_mask.select(1, k), B) / (double)K;
      latent = dyn.latent;
    }
  }
  t::Tensor loss = c.policy_weight * lp + c.value_weight * lv + c.reward_weight * lr_ +
                   c.consistency_weight * lc + c.reconstruction_weight * lg + c.ownership_weight * lo;
  if (autocast){
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    at::autocast::clear_cache();
  }

  mAdam->zero_grad();
  if (mMuon)
    for (t::Tensor& p : mMuon->params) p.mutable_grad() = t::Tensor();
  loss.backward();
  double grad_norm = t::nn::utils::clip_grad_norm_(model->parameters(), c.grad_clip);
  mAdam->step();
  if (mMuon) mMuon->step(c.muon_learning_rate * mult);
  ++mStep;

  t::Tensor all = t::stack({loss, lp, lv, lr_, lc, lg, lo}).to(t::kCPU, t::kFloat);
  const float* v = all.data_ptr<float>();
  LearnerStats st;
  st.step = mStep;
  st.loss = v[0];
  st.policy = v[1];
  st.value = v[2];
  st.reward = v[3];
  st.consistency = v[4];
  st.reconstruction = v[5];
  st.ownership = v[6];
  st.grad_norm = grad_norm;
  st.learning_rate = lr;
  return st;
}

void Learner::save_weights(const s::string& file){
  DeviceStream::Scope scope(mStream);
  mStream.synchronize();
  write_atomically(file, [&](const s::string& p){ save_model(mModel, p); });
}

TransformerMuZero Learner::copy_model(t::ScalarType dtype){
  DeviceStream::Scope scope(mStream);
  TransformerMuZero copy(mModel->config);
  copy->to(mDevice, dtype);
  copy_weights(copy, mModel);
  mStream.synchronize();
  return copy;
}

void Learner::save_checkpoint(const s::string& dir){
  DeviceStream::Scope scope(mStream);
  mStream.synchronize();
  sf::path d(dir);
  sf::create_directories(d);
  write_atomically(d / "model.pt", [&](const s::string& p){ save_model(mModel, p); });
  write_atomically(d / "optimizer.pt", [&](const s::string& p){
    t::serialize::OutputArchive archive;
    t::serialize::OutputArchive adam;
    mAdam->save(adam);
    archive.write("adamw", adam);
    if (mMuon){
      t::serialize::OutputArchive muon;
      mMuon->save(muon);
      archive.write("muon", muon);
    }
    archive.save_to(p);
  });
  write_atomically(d / "learner.txt", [&](const s::string& p){ s::ofstream(p) << mStep << "\n"; });
}

bool Learner::load_checkpoint(const s::string& dir){
  sf::path d(dir);
  if (not sf::exists(d / "model.pt") or not sf::exists(d / "learner.txt")) return false;
  DeviceStream::Scope scope(mStream);
  TransformerMuZero loaded = load_model((d / "model.pt").string(), t::kCPU);
  copy_weights(mModel, loaded);
  if (sf::exists(d / "optimizer.pt")){
    t::serialize::InputArchive archive;
    archive.load_from((d / "optimizer.pt").string(), mDevice);
    t::serialize::InputArchive adam;
    archive.read("adamw", adam);
    mAdam->load(adam);
    t::serialize::InputArchive muon;
    if (mMuon and archive.try_read("muon", muon)) mMuon->load(muon);
  }
  s::ifstream(d / "learner.txt") >> mStep;
  mStream.synchronize();
  return true;
}

}  // namespace rlbg::libtorch
