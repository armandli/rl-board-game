#include <libtorch/transformer_muzero.h>

#include <format>
#include <stdexcept>

namespace rlbg::libtorch {

s::vector<int64_t> TransformerMuZeroConfig::to_vector() const {
  return {num_tokens, num_features, num_actions, num_token_classes,
          dims, num_heads, repr_layers, dyn_layers, pred_layers, proj_dims};
}

TransformerMuZeroConfig TransformerMuZeroConfig::from_vector(const s::vector<int64_t>& v){
  if (v.size() != 10) throw s::invalid_argument("bad TransformerMuZeroConfig");
  TransformerMuZeroConfig c;
  c.num_tokens = v[0];
  c.num_features = v[1];
  c.num_actions = v[2];
  c.num_token_classes = v[3];
  c.dims = v[4];
  c.num_heads = v[5];
  c.repr_layers = v[6];
  c.dyn_layers = v[7];
  c.pred_layers = v[8];
  c.proj_dims = v[9];
  return c;
}

s::string TransformerMuZeroConfig::describe() const {
  return s::format("dims {} heads {} layers {}/{}/{} (repr/dyn/pred)", dims, num_heads, repr_layers, dyn_layers,
                   pred_layers);
}

RMSNormImpl::RMSNormImpl(int64_t dims, double eps)
: eps(eps)
{
  weight = register_parameter("weight", t::ones({dims}));
}

t::Tensor RMSNormImpl::forward(const t::Tensor& x){
  // the fused kernel accumulates in float32 and keeps no float32 copies of
  // x for the backward pass, unlike the composite formula
  return t::rms_norm(x, {weight.size(0)}, weight.to(x.scalar_type()), eps);
}

AttentionImpl::AttentionImpl(int64_t dims, int64_t heads)
: heads(heads)
{
  qkv = register_module("qkv", t::nn::Linear(t::nn::LinearOptions(dims, 3 * dims).bias(false)));
  out = register_module("out", t::nn::Linear(t::nn::LinearOptions(dims, dims).bias(false)));
}

t::Tensor AttentionImpl::forward(const t::Tensor& x){
  int64_t b = x.size(0), n = x.size(1), d = x.size(2);
  t::Tensor q_k_v = qkv(x).view({b, n, 3, heads, d / heads}).permute({2, 0, 3, 1, 4});
  t::Tensor y = at::scaled_dot_product_attention(q_k_v[0], q_k_v[1], q_k_v[2]);
  return out(y.transpose(1, 2).reshape({b, n, d}));
}

SwiGLUImpl::SwiGLUImpl(int64_t dims){
  // 8/3 of the width, rounded up to a multiple of 32
  int64_t hidden = ((dims * 8 / 3) + 31) / 32 * 32;
  w1 = register_module("w1", t::nn::Linear(t::nn::LinearOptions(dims, hidden).bias(false)));
  w3 = register_module("w3", t::nn::Linear(t::nn::LinearOptions(dims, hidden).bias(false)));
  w2 = register_module("w2", t::nn::Linear(t::nn::LinearOptions(hidden, dims).bias(false)));
}

t::Tensor SwiGLUImpl::forward(const t::Tensor& x){
  return w2(t::silu(w1(x)) * w3(x));
}

BlockImpl::BlockImpl(int64_t dims, int64_t heads){
  norm1 = register_module("norm1", RMSNorm(dims));
  attn = register_module("attn", Attention(dims, heads));
  norm2 = register_module("norm2", RMSNorm(dims));
  mlp = register_module("mlp", SwiGLU(dims));
}

t::Tensor BlockImpl::forward(t::Tensor x){
  x = x + attn(norm1(x));
  return x + mlp(norm2(x));
}

TransformerMuZeroImpl::TransformerMuZeroImpl(const TransformerMuZeroConfig& config)
: config(config)
{
  if (config.dims % config.num_heads != 0)
    throw s::invalid_argument("dims must be divisible by num_heads");
  if (config.num_actions != config.num_tokens + 1)
    throw s::invalid_argument("expects one action per token plus pass");
  if (config.num_features <= config.num_token_classes)
    throw s::invalid_argument("expects token class one-hot features followed by others");
  int64_t d = config.dims;
  int64_t c = config.num_token_classes;
  int64_t p = config.proj_dims;
  auto plain_norm = [](int64_t n){ return t::nn::LayerNorm(t::nn::LayerNormOptions({n}).elementwise_affine(false)); };

  class_embedding = register_module("class_embedding", t::nn::Embedding(c, d));
  feature_proj = register_module("feature_proj", t::nn::Linear(config.num_features - c, d));
  pos_embedding = register_parameter("pos_embedding", t::randn({config.num_tokens, d}) * 0.02);
  repr_blocks = register_module("repr_blocks", make_blocks(config.repr_layers));
  repr_norm = register_module("repr_norm", plain_norm(d));

  action_embedding = register_module("action_embedding", t::nn::Embedding(config.num_actions, d));
  action_marker = register_parameter("action_marker", t::randn({d}) * 0.02);
  dyn_blocks = register_module("dyn_blocks", make_blocks(config.dyn_layers));
  dyn_norm = register_module("dyn_norm", plain_norm(d));
  reward_norm = register_module("reward_norm", RMSNorm(d));
  reward_head = register_module("reward_head", t::nn::Linear(d, NUM_OUTCOME_CLASSES));

  pred_blocks = register_module("pred_blocks", make_blocks(config.pred_layers));
  pred_norm = register_module("pred_norm", RMSNorm(d));
  policy_head = register_module("policy_head", t::nn::Linear(d, 1));
  pass_head = register_module("pass_head", t::nn::Linear(d, 1));
  value_head = register_module("value_head", t::nn::Sequential(
    t::nn::Linear(d, d), t::nn::SiLU(), t::nn::Linear(d, NUM_OUTCOME_CLASSES)));
  ownership_head = register_module("ownership_head", t::nn::Linear(d, c));

  recon_head = register_module("recon_head", t::nn::Linear(d, c));
  projector = register_module("projector", t::nn::Sequential(
    t::nn::Linear(d, p), t::nn::LayerNorm(t::nn::LayerNormOptions({p})), t::nn::ReLU(), t::nn::Linear(p, p)));
  predictor = register_module("predictor", t::nn::Sequential(
    t::nn::Linear(p, p), t::nn::LayerNorm(t::nn::LayerNormOptions({p})), t::nn::ReLU(), t::nn::Linear(p, p)));
}

t::nn::ModuleList TransformerMuZeroImpl::make_blocks(int64_t layers){
  t::nn::ModuleList blocks;
  for (int64_t i = 0; i < layers; ++i)
    blocks->push_back(Block(config.dims, config.num_heads));
  return blocks;
}

t::Tensor TransformerMuZeroImpl::run_blocks(t::nn::ModuleList& blocks, t::Tensor x){
  for (auto& m : *blocks) x = m->as<Block>()->forward(x);
  return x;
}

t::Tensor TransformerMuZeroImpl::represent(const t::Tensor& features){
  int64_t c = config.num_token_classes;
  // class index from its one-hot features
  t::Tensor onehot = features.slice(-1, 0, c).to(t::kLong);
  t::Tensor cls = (onehot * t::arange(c, onehot.options())).sum(-1);
  t::Tensor rest = features.slice(-1, c).to(pos_embedding.scalar_type());
  t::Tensor x = class_embedding(cls) + feature_proj(rest) + pos_embedding;
  return repr_norm(run_blocks(repr_blocks, x));
}

DynamicsOutput TransformerMuZeroImpl::dynamics(const t::Tensor& latent, const t::Tensor& actions){
  int64_t n = latent.size(1);
  t::Tensor positions = t::arange(n, actions.options());
  // pass (index n) marks no square
  t::Tensor onehot = actions.unsqueeze(1).eq(positions.unsqueeze(0)).to(latent.scalar_type());
  t::Tensor tokens = latent + onehot.unsqueeze(-1) * action_marker.to(latent.scalar_type());
  t::Tensor action_token = action_embedding(actions).unsqueeze(1).to(latent.scalar_type());
  t::Tensor x = run_blocks(dyn_blocks, t::cat({action_token, tokens}, 1));
  t::Tensor head = x.select(1, 0);
  t::Tensor next = x.slice(1, 1);
  return {dyn_norm(next), reward_head(reward_norm(head))};
}

Prediction TransformerMuZeroImpl::predict(const t::Tensor& latent){
  t::Tensor h = pred_norm(run_blocks(pred_blocks, latent));
  t::Tensor pooled = h.mean(1);
  t::Tensor policy = t::cat({policy_head(h).squeeze(-1), pass_head(pooled)}, 1);
  return {policy, value_head->forward(pooled), ownership_head(h)};
}

t::Tensor TransformerMuZeroImpl::project(const t::Tensor& latent){
  return projector->forward(latent);
}

t::Tensor TransformerMuZeroImpl::predict_projection(const t::Tensor& projection){
  return predictor->forward(projection);
}

t::Tensor TransformerMuZeroImpl::reconstruct(const t::Tensor& latent){
  return recon_head(latent);
}

int64_t TransformerMuZeroImpl::num_parameters() const {
  int64_t n = 0;
  for (const auto& p : parameters()) n += p.numel();
  return n;
}

void copy_weights(TransformerMuZero& dst, TransformerMuZero& src){
  if (not (dst->config == src->config)) throw s::invalid_argument("copy_weights: model configurations differ");
  t::NoGradGuard no_grad;
  auto dp = dst->named_parameters(true);
  for (const auto& item : src->named_parameters(true)) dp[item.key()].copy_(item.value());
  auto db = dst->named_buffers(true);
  for (const auto& item : src->named_buffers(true)) db[item.key()].copy_(item.value());
}

void save_model(TransformerMuZero& model, const s::string& path){
  t::serialize::OutputArchive archive;
  archive.write("config", t::tensor(model->config.to_vector(), t::kLong));
  t::serialize::OutputArchive weights;
  model->save(weights);
  archive.write("weights", weights);
  archive.save_to(path);
}

namespace {

TransformerMuZeroConfig read_config(t::serialize::InputArchive& archive){
  t::Tensor cfg;
  archive.read("config", cfg);
  cfg = cfg.to(t::kCPU).contiguous();
  return TransformerMuZeroConfig::from_vector(s::vector<int64_t>(cfg.data_ptr<int64_t>(), cfg.data_ptr<int64_t>() + cfg.numel()));
}

} // namespace

TransformerMuZero load_model(const s::string& path, t::Device device){
  t::serialize::InputArchive archive;
  archive.load_from(path, t::kCPU);
  TransformerMuZero model(read_config(archive));
  t::serialize::InputArchive weights;
  archive.read("weights", weights);
  model->load(weights);
  model->to(device);
  return model;
}

TransformerMuZeroConfig load_model_config(const s::string& path){
  t::serialize::InputArchive archive;
  archive.load_from(path, t::kCPU);
  return read_config(archive);
}

}  // namespace rlbg::libtorch
