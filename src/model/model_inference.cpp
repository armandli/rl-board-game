#include <model/model_inference.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rlbg {

mx::Dtype parse_dtype(const s::string& name) {
  if (name == "float32") return mx::float32;
  if (name == "float16") return mx::float16;
  if (name == "bfloat16") return mx::bfloat16;
  throw s::invalid_argument("unknown dtype " + name);
}

ModelInference::ModelInference(s::shared_ptr<MuZeroModel> model, int observation_size, mx::Dtype dtype)
  : mModel(s::move(model)), mObservationSize(observation_size), mDtype(dtype)
{
  mModel->eval();
  if (mDtype != mx::float32) mModel->set_dtype(mDtype);
  // evaluated here so another thread can run the model (MLX binds lazy
  // arrays to the creating thread's stream)
  nn::eval(mModel->parameters());
}

void ModelInference::set_weights(const nn::Params& weights) {
  s::lock_guard<s::mutex> lock(mMutex);
  mModel->update(weights);
  if (mDtype != mx::float32) mModel->set_dtype(mDtype);
  nn::eval(mModel->parameters());
}

void ModelInference::infer(const InferenceRequests& requests, InferenceResults& results) {
  s::lock_guard<s::mutex> lock(mMutex);
  int ni = requests.num_initial();
  int nr = requests.num_recurrent();
  int t = mModel->num_tokens();
  int d = mModel->latent_dims();
  int a = mModel->num_actions();

  results.num_initial = ni;
  results.latent_size = t * d;
  results.num_actions = a;
  int n = ni + nr;
  results.latents.resize((size_t)n * t * d);
  results.policy_logits.resize((size_t)n * a);
  results.values.resize(n);
  results.rewards.assign(n, 0.f);
  if (n == 0) return;

  s::vector<mx::array> latents;
  s::vector<mx::array> rewards;
  if (ni > 0) {
    mx::array feat(requests.features.begin(), {ni, t, mObservationSize / t}, mx::uint8);
    latents.push_back(mModel->represent(mx::astype(feat, mDtype)));
    rewards.push_back(mx::zeros({ni}, mx::float32));
  }
  if (nr > 0) {
    mx::array lat(requests.latents.begin(), {nr, t, d}, mx::float16);
    mx::array act(requests.actions.begin(), {nr}, mx::int32);
    DynamicsOutput dyn = mModel->dynamics(mx::astype(lat, mDtype), act);
    latents.push_back(dyn.latent);
    rewards.push_back(mx::astype(outcome_scalar(dyn.reward_logits), mx::float32));
  }
  mx::array latent = latents.size() == 1 ? latents[0] : mx::concatenate(latents, 0);
  Prediction pred = mModel->predict(latent);

  mx::array lat16 = mx::contiguous(mx::astype(latent, mx::float16));
  mx::array logits = mx::contiguous(mx::astype(pred.policy_logits, mx::float32));
  mx::array value = mx::contiguous(mx::astype(outcome_scalar(pred.value_logits), mx::float32));
  mx::array reward = mx::contiguous(rewards.size() == 1 ? rewards[0] : mx::concatenate(rewards, 0));
  mx::eval({lat16, logits, value, reward});

  s::memcpy(results.latents.data(), lat16.data<mx::float16_t>(), results.latents.size() * sizeof(latent_t));
  s::memcpy(results.policy_logits.data(), logits.data<float>(), results.policy_logits.size() * sizeof(float));
  s::memcpy(results.values.data(), value.data<float>(), n * sizeof(float));
  s::memcpy(results.rewards.data(), reward.data<float>(), n * sizeof(float));
}

}  // namespace rlbg
