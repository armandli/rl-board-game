#include <training/muzero_learner.h>

#include <model/model_inference.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rlbg {

namespace sf = std::filesystem;

namespace {

// masked sum over the batch divided by the batch size
mx::array masked_mean(const mx::array& loss, const mx::array& mask, int batch) {
  return mx::sum(loss * mask) / mx::array((float)batch);
}

// column k of a (B, K, ...) array
mx::array at_step(const mx::array& x, int k) {
  return mx::take(x, k, 1);
}

} // namespace

MuZeroLearner::MuZeroLearner(s::shared_ptr<MuZeroModel> model, const MuZeroLearnerConfig& config)
  : mModel(s::move(model)), mConfig(config),
    mCompute(parse_dtype(config.compute_dtype))
{
  mModel->train();
  auto adamw = s::make_shared<optim::AdamW>(
    schedule(config.learning_rate), s::pair<double, double>{0.9, 0.95}, 1e-8, config.weight_decay
  );
  if (config.optimizer == "adamw") {
    mOptimizer = adamw;
  } else if (config.optimizer == "muon") {
    auto muon = s::make_shared<optim::Muon>(schedule(config.muon_learning_rate), 0.95, config.weight_decay);
    // hidden weight matrices of the transformer blocks; embeddings, heads,
    // norms and biases stay with AdamW
    optim::MultiOptimizer::Filter hidden = [](const s::string& key, const mx::array& g) {
      return g.ndim() == 2 and key.find("_blocks.") != s::string::npos;
    };
    mOptimizer = s::make_shared<optim::MultiOptimizer>(
      s::vector<s::shared_ptr<optim::Optimizer>>{muon, adamw},
      s::vector<optim::MultiOptimizer::Filter>{hidden}
    );
  } else {
    throw s::invalid_argument("unknown optimizer " + config.optimizer);
  }
  // the scheduled learning rate starts as a lazy array; evaluate it so the
  // learner can train on another thread than the one that built it
  nn::eval(mOptimizer->state());
}

optim::Schedule MuZeroLearner::schedule(double peak) const {
  int warmup = s::max(1, mConfig.warmup_steps);
  int decay = s::max(1, mConfig.total_steps - warmup);
  return optim::join_schedules(
    {optim::linear_schedule(peak * 0.01, peak, warmup),
     optim::cosine_decay(peak, decay, peak * mConfig.final_lr_fraction)},
    {warmup}
  );
}

LearnerStats MuZeroLearner::train_step(const TrainingBatch& batch) {
  const int B = batch.B, K = batch.K, T = batch.T, F = batch.F, A = batch.A;
  MuZeroModel& model = *mModel;

  mx::array obs = mx::astype(mx::array(batch.observations.begin(), {B, T, F}, mx::uint8), mx::float32);
  mx::array target_obs = mx::astype(mx::array(batch.target_observations.begin(), {B * K, T, F}, mx::uint8), mx::float32);
  mx::array actions(batch.actions.begin(), {B, K}, mx::int32);
  mx::array policies(batch.policies.begin(), {B, K + 1, A}, mx::float32);
  mx::array policy_mask(batch.policy_mask.begin(), {B, K + 1}, mx::float32);
  mx::array values(batch.values.begin(), {B, K + 1}, mx::int32);
  mx::array value_mask(batch.value_mask.begin(), {B, K + 1}, mx::float32);
  mx::array rewards(batch.rewards.begin(), {B, K}, mx::int32);
  mx::array reward_mask(batch.reward_mask.begin(), {B, K}, mx::float32);
  mx::array tokens(batch.tokens.begin(), {B, K + 1, T}, mx::int32);
  mx::array ownership(batch.ownership.begin(), {B, K + 1, T}, mx::int32);
  mx::array state_mask(batch.state_mask.begin(), {B, K + 1}, mx::float32);

  s::optional<mx::array> policy_loss, value_loss, reward_loss, consistency_loss, recon_loss, ownership_loss;

  auto f32 = [](const mx::array& x) { return mx::astype(x, mx::float32); };

  auto loss_fn = [&]() {
    // mixed precision: the traced float32 parameters are cast to the compute
    // dtype inside the trace, so gradients still arrive in float32
    if (mCompute != mx::float32) model.set_dtype(mCompute);
    mx::array zero = mx::array(0.f);
    mx::array lp = zero, lv = zero, lr = zero, lc = zero, lg = zero, lo = zero;

    // consistency targets: projections of the real future states, no gradient
    mx::array target_proj = mx::stop_gradient(f32(model.project(model.represent(target_obs))));
    target_proj = mx::reshape(target_proj, {B, K, -1});
    target_proj = target_proj / (mx::sqrt(mx::sum(mx::square(target_proj), -1, true)) + mx::array(1e-6f));

    mx::array latent = model.represent(obs);
    for (int k = 0; k <= K; ++k) {
      mx::array w = mx::array(k == 0 ? 1.f : 1.f / K);
      Prediction pred = model.predict(latent);

      mx::array pce = nn::cross_entropy(f32(pred.policy_logits), at_step(policies, k));
      lp = lp + w * masked_mean(pce, at_step(policy_mask, k), B);

      mx::array vce = nn::cross_entropy(f32(pred.value_logits), at_step(values, k));
      lv = lv + w * masked_mean(vce, at_step(value_mask, k), B);

      mx::array oce = mx::mean(nn::cross_entropy(f32(pred.ownership_logits), at_step(ownership, k)), -1);
      lo = lo + w * masked_mean(oce, at_step(value_mask, k), B);

      mx::array gce = mx::mean(nn::cross_entropy(f32(model.reconstruct(latent)), at_step(tokens, k)), -1);
      lg = lg + w * masked_mean(gce, at_step(state_mask, k), B);

      if (k > 0) {
        mx::array p = mx::reshape(f32(model.predict_projection(model.project(latent))), {B, -1});
        p = p / (mx::sqrt(mx::sum(mx::square(p), -1, true)) + mx::array(1e-6f));
        mx::array cos = mx::sum(p * at_step(target_proj, k - 1), -1);
        lc = lc + masked_mean(-cos, at_step(state_mask, k), B) / mx::array((float)K);
      }
      if (k < K) {
        DynamicsOutput dyn = model.dynamics(
          nn::scale_gradient(latent, mConfig.dynamics_gradient_scale), at_step(actions, k)
        );
        mx::array rce = nn::cross_entropy(f32(dyn.reward_logits), at_step(rewards, k));
        lr = lr + masked_mean(rce, at_step(reward_mask, k), B) / mx::array((float)K);
        latent = dyn.latent;
      }
    }
    policy_loss = lp;
    value_loss = lv;
    reward_loss = lr;
    consistency_loss = lc;
    recon_loss = lg;
    ownership_loss = lo;
    return lp * mx::array((float)mConfig.policy_weight) +
           lv * mx::array((float)mConfig.value_weight) +
           lr * mx::array((float)mConfig.reward_weight) +
           lc * mx::array((float)mConfig.consistency_weight) +
           lg * mx::array((float)mConfig.reconstruction_weight) +
           lo * mx::array((float)mConfig.ownership_weight);
  };

  auto [loss, grads] = nn::compute_value_and_grad(model, loss_fn);
  auto [clipped, grad_norm] = optim::clip_grad_norm(grads, mConfig.grad_clip);
  mx::array lr = mOptimizer->learning_rate();
  mOptimizer->update(model, clipped);

  s::vector<mx::array> outputs{
    loss, *policy_loss, *value_loss, *reward_loss, *consistency_loss, *recon_loss, *ownership_loss, grad_norm, lr
  };
  for (const auto& [k, v] : model.parameters()) outputs.push_back(v);
  for (const auto& [k, v] : mOptimizer->state()) outputs.push_back(v);
  mx::eval(outputs);
  ++mStep;

  LearnerStats st;
  st.step = mStep;
  st.loss = loss.item<float>();
  st.policy = policy_loss->item<float>();
  st.value = value_loss->item<float>();
  st.reward = reward_loss->item<float>();
  st.consistency = consistency_loss->item<float>();
  st.reconstruction = recon_loss->item<float>();
  st.ownership = ownership_loss->item<float>();
  st.grad_norm = grad_norm.item<float>();
  st.learning_rate = mx::astype(lr, mx::float32).item<float>();
  return st;
}

nn::Params MuZeroLearner::weights() const {
  return mModel->parameters();
}

void MuZeroLearner::save_checkpoint(const s::string& dir) const {
  sf::create_directories(dir);
  // write then rename so a crash never leaves a half written checkpoint
  mModel->save_weights(dir + "/model.tmp.safetensors");
  mOptimizer->save_state(dir + "/optimizer.tmp.safetensors");
  sf::rename(dir + "/model.tmp.safetensors", dir + "/model.safetensors");
  sf::rename(dir + "/optimizer.tmp.safetensors", dir + "/optimizer.safetensors");
  s::ofstream(dir + "/learner.txt") << mStep << "\n";
}

bool MuZeroLearner::load_checkpoint(const s::string& dir) {
  if (not sf::exists(dir + "/model.safetensors") or not sf::exists(dir + "/learner.txt")) return false;
  mModel->load_weights(dir + "/model.safetensors");
  if (sf::exists(dir + "/optimizer.safetensors")) mOptimizer->load_state(dir + "/optimizer.safetensors");
  s::ifstream(dir + "/learner.txt") >> mStep;
  // loaded arrays are lazy; evaluate them on this thread (see constructor)
  nn::eval(mModel->parameters());
  nn::eval(mOptimizer->state());
  return true;
}

}  // namespace rlbg
