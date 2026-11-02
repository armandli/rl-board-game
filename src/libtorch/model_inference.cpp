#include <libtorch/model_inference.h>

#include <cstring>
#include <format>
#include <stdexcept>

namespace rlbg::libtorch {

static_assert(sizeof(latent_t) == sizeof(c10::Half));
static_assert(sizeof(uint64_t) % sizeof(latent_t) == 0);

t::ScalarType parse_dtype(const s::string& name){
  if (name == "float32") return t::kFloat;
  if (name == "float16") return t::kHalf;
  if (name == "bfloat16") return t::kBFloat16;
  throw s::invalid_argument("unknown dtype " + name);
}

ModelInference::ModelInference(const TransformerMuZeroConfig& config, t::Device device, t::ScalarType dtype,
                               int64_t latent_pool)
: mConfig(config), mModel(config), mDevice(device), mDtype(dtype), mStream(device), mPoolCapacity(latent_pool)
{
  DeviceStream::Scope scope(mStream);
  mModel->to(device, dtype);
  mModel->eval();
  if (mPoolCapacity > 0)
    mPool = t::zeros({mPoolCapacity, config.num_tokens, config.dims}, t::TensorOptions().device(device).dtype(dtype));
  mStream.synchronize();
}

void ModelInference::load_weights(const s::string& file){
  TransformerMuZero loaded = load_model(file, t::kCPU);
  set_weights(loaded);
}

void ModelInference::set_weights(TransformerMuZero& src){
  s::lock_guard<s::mutex> lock(mMutex);
  DeviceStream::Scope scope(mStream);
  copy_weights(mModel, src);
  // src may be freed (and its memory reused by another stream) once this
  // returns
  mStream.synchronize();
}

t::Tensor ModelInference::host_buffer(t::Tensor& buffer, int64_t numel, t::ScalarType type){
  if (not buffer.defined() or buffer.numel() < numel)
    buffer = t::empty({s::max<int64_t>(numel, buffer.defined() ? 2 * buffer.numel() : 0)},
                      t::TensorOptions().dtype(type).pinned_memory(mStream.pinned()));
  return buffer.slice(0, 0, numel);
}

t::Tensor ModelInference::input_latents(const InferenceRequests& requests){
  int nr = requests.num_recurrent();
  int64_t T = mConfig.num_tokens, D = mConfig.dims;
  bool async = mStream.pinned();
  if (mPoolCapacity == 0){
    t::Tensor lat = host_buffer(mLatentsIn, (int64_t)nr * T * D, t::kHalf);
    s::memcpy(lat.data_ptr(), requests.latents.data(), (size_t)nr * T * D * sizeof(latent_t));
    return lat.view({nr, T, D}).to(mDevice, mDtype, async);
  }
  t::Tensor slots = host_buffer(mSlots, nr, t::kLong);
  int64_t* sp = slots.data_ptr<int64_t>();
  for (int i = 0; i < nr; ++i){
    uint64_t handle;
    s::memcpy(&handle, requests.latents.data() + (size_t)i * LATENT_HANDLE_SIZE, sizeof(handle));
    if (handle >= mNextHandle or mNextHandle - handle > (uint64_t)mPoolCapacity)
      throw s::runtime_error(s::format(
        "latent {} is no longer in the pool of {} (next {}); raise the latent pool size", handle, mPoolCapacity,
        mNextHandle));
    sp[i] = (int64_t)(handle % mPoolCapacity);
  }
  return mPool.index_select(0, slots.to(mDevice, async));
}

void ModelInference::output_latents(const t::Tensor& latent, InferenceResults& results, bool async){
  int64_t n = latent.size(0), T = mConfig.num_tokens, D = mConfig.dims;
  if (mPoolCapacity == 0){
    t::Tensor out = host_buffer(mLatentsOut, n * T * D, t::kHalf);
    out.copy_(latent.to(t::kHalf).reshape({-1}), async);
    return;
  }
  if (n > mPoolCapacity) throw s::runtime_error("inference batch larger than the latent pool");
  int64_t start = (int64_t)(mNextHandle % mPoolCapacity);
  int64_t first = s::min(n, mPoolCapacity - start);
  mPool.slice(0, start, start + first).copy_(latent.slice(0, 0, first));
  if (first < n) mPool.slice(0, 0, n - first).copy_(latent.slice(0, first));
  for (int64_t i = 0; i < n; ++i){
    uint64_t handle = mNextHandle + i;
    s::memcpy(results.latents.data() + (size_t)i * LATENT_HANDLE_SIZE, &handle, sizeof(handle));
  }
  mNextHandle += n;
}

void ModelInference::infer(const InferenceRequests& requests, InferenceResults& results){
  s::lock_guard<s::mutex> lock(mMutex);
  int ni = requests.num_initial();
  int nr = requests.num_recurrent();
  int n = ni + nr;
  int64_t T = mConfig.num_tokens, F = mConfig.num_features, A = mConfig.num_actions;
  int lsize = latent_size();
  results.num_initial = ni;
  results.latent_size = lsize;
  results.num_actions = (int)A;
  results.latents.resize((size_t)n * lsize);
  results.policy_logits.resize((size_t)n * A);
  results.values.resize(n);
  results.rewards.resize(n);
  if (n == 0) return;

  DeviceStream::Scope scope(mStream);
  t::InferenceMode guard;
  bool async = mStream.pinned();
  s::vector<t::Tensor> latents;
  s::vector<t::Tensor> rewards;
  if (ni > 0){
    t::Tensor feat = host_buffer(mFeatures, (int64_t)ni * T * F, t::kUInt8);
    s::memcpy(feat.data_ptr(), requests.features.data(), (size_t)ni * T * F);
    latents.push_back(mModel->represent(feat.view({ni, T, F}).to(mDevice, async)));
    rewards.push_back(t::zeros({ni}, t::TensorOptions().device(mDevice).dtype(t::kFloat)));
  }
  if (nr > 0){
    t::Tensor act = host_buffer(mActions, nr, t::kLong);
    int64_t* ap = act.data_ptr<int64_t>();
    for (int i = 0; i < nr; ++i) ap[i] = requests.actions[i];
    DynamicsOutput dyn = mModel->dynamics(input_latents(requests), act.to(mDevice, async));
    latents.push_back(dyn.latent);
    rewards.push_back(outcome_scalar(dyn.reward_logits));
  }
  t::Tensor latent = latents.size() == 1 ? latents[0] : t::cat(latents, 0);
  Prediction pred = mModel->predict(latent);

  // policy logits, value and reward of every position in one transfer
  t::Tensor scalars = t::cat({pred.policy_logits.to(t::kFloat), outcome_scalar(pred.value_logits).unsqueeze(1),
                              (rewards.size() == 1 ? rewards[0] : t::cat(rewards, 0)).unsqueeze(1)}, 1);
  t::Tensor sc_out = host_buffer(mScalarsOut, (int64_t)n * (A + 2), t::kFloat);
  sc_out.copy_(scalars.reshape({-1}), async);
  output_latents(latent, results, async);
  mStream.synchronize();

  if (mPoolCapacity == 0)
    s::memcpy(results.latents.data(), mLatentsOut.data_ptr(), results.latents.size() * sizeof(latent_t));
  const float* sc = sc_out.data_ptr<float>();
  for (int i = 0; i < n; ++i){
    const float* row = sc + (size_t)i * (A + 2);
    s::memcpy(results.policy_logits.data() + (size_t)i * A, row, A * sizeof(float));
    results.values[i] = row[A];
    results.rewards[i] = row[A + 1];
  }
}

}  // namespace rlbg::libtorch
