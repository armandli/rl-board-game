#include <model/transformer_muzero.h>

#include <stdexcept>

namespace rlbg {

namespace {

s::shared_ptr<nn::ModuleList> make_blocks(int layers, int dims, int heads) {
  auto blocks = s::make_shared<nn::ModuleList>();
  for (int i = 0; i < layers; ++i)
    blocks->append(s::make_shared<nn::TransformerBlock>(dims, heads));
  return blocks;
}

} // namespace

TransformerMuZero::TransformerMuZero(const TransformerMuZeroConfig& config)
  : config(config),
    pos_embedding(mx::random::normal({config.num_tokens, config.dims}) * mx::array(0.02f)),
    action_marker(mx::random::normal({config.dims}) * mx::array(0.02f))
{
  if (config.dims % config.num_heads != 0)
    throw s::invalid_argument("dims must be divisible by num_heads");
  int d = config.dims;
  int c = config.num_token_classes;

  input_proj = register_module("input_proj", s::make_shared<nn::Linear>(config.num_features, d));
  register_parameter("pos_embedding", pos_embedding);
  repr_blocks = register_module("repr_blocks", make_blocks(config.repr_layers, d, config.num_heads));
  repr_norm = register_module("repr_norm", s::make_shared<nn::LayerNorm>(d, 1e-5f, false));

  action_embedding = register_module("action_embedding", s::make_shared<nn::Embedding>(config.num_actions, d));
  register_parameter("action_marker", action_marker);
  dyn_blocks = register_module("dyn_blocks", make_blocks(config.dyn_layers, d, config.num_heads));
  dyn_norm = register_module("dyn_norm", s::make_shared<nn::LayerNorm>(d, 1e-5f, false));
  reward_norm = register_module("reward_norm", s::make_shared<nn::RMSNorm>(d));
  reward_head = register_module("reward_head", s::make_shared<nn::Linear>(d, NUM_OUTCOME_CLASSES));

  pred_blocks = register_module("pred_blocks", make_blocks(config.pred_layers, d, config.num_heads));
  pred_norm = register_module("pred_norm", s::make_shared<nn::RMSNorm>(d));
  if (per_square_policy()) {
    policy_head = register_module("policy_head", s::make_shared<nn::Linear>(d, 1));
    pass_head = register_module("pass_head", s::make_shared<nn::Linear>(d, 1));
  } else {
    policy_head = register_module("policy_head", s::make_shared<nn::Linear>(d, config.num_actions));
  }
  value_head = register_module("value_head", s::make_shared<nn::Sequential>(
    s::make_shared<nn::Linear>(d, d),
    nn::silu,
    s::make_shared<nn::Linear>(d, NUM_OUTCOME_CLASSES)
  ));
  ownership_head = register_module("ownership_head", s::make_shared<nn::Linear>(d, c));

  recon_head = register_module("recon_head", s::make_shared<nn::Linear>(d, c));
  int p = config.proj_dims;
  projector = register_module("projector", s::make_shared<nn::Sequential>(
    s::make_shared<nn::Linear>(d, p),
    s::make_shared<nn::LayerNorm>(p),
    nn::relu,
    s::make_shared<nn::Linear>(p, p)
  ));
  predictor = register_module("predictor", s::make_shared<nn::Sequential>(
    s::make_shared<nn::Linear>(p, p),
    s::make_shared<nn::LayerNorm>(p),
    nn::relu,
    s::make_shared<nn::Linear>(p, p)
  ));
  // MLX binds a lazy array to the creating thread's stream; materialize the
  // initial weights so the model can be used from any thread
  nn::eval(parameters());
}

mx::array TransformerMuZero::run_blocks(const nn::ModuleList& blocks, mx::array x) {
  for (size_t i = 0; i < blocks.size(); ++i)
    x = blocks.at<nn::TransformerBlock>(i)->forward(x);
  return x;
}

mx::array TransformerMuZero::represent(const mx::array& features) {
  mx::array x = mx::astype(features, input_proj->weight.dtype());
  x = input_proj->forward(x) + pos_embedding;
  return repr_norm->forward(run_blocks(*repr_blocks, x));
}

DynamicsOutput TransformerMuZero::dynamics(const mx::array& latent, const mx::array& actions) {
  int b = latent.shape(0);
  int t = latent.shape(1);
  int d = latent.shape(2);
  mx::Dtype dt = latent.dtype();
  // pass (or any non-square action) marks no square
  mx::array onehot = mx::astype(
    mx::equal(mx::expand_dims(actions, 1), mx::expand_dims(mx::arange(t), 0)), dt
  );
  mx::array tokens = latent + mx::expand_dims(onehot, -1) * action_marker;
  mx::array action_token = mx::expand_dims(action_embedding->forward(actions), 1);
  mx::array x = mx::concatenate({action_token, tokens}, 1);
  x = run_blocks(*dyn_blocks, x);
  mx::array head = mx::reshape(mx::slice(x, {0, 0, 0}, {b, 1, d}), {b, d});
  mx::array next = mx::slice(x, {0, 1, 0}, {b, t + 1, d});
  return {dyn_norm->forward(next), reward_head->forward(reward_norm->forward(head))};
}

Prediction TransformerMuZero::predict(const mx::array& latent) {
  mx::array h = pred_norm->forward(run_blocks(*pred_blocks, latent));
  mx::array pooled = mx::mean(h, 1);
  mx::array policy = per_square_policy()
    ? mx::concatenate({mx::squeeze(policy_head->forward(h), -1), pass_head->forward(pooled)}, 1)
    : policy_head->forward(pooled);
  return {policy, value_head->forward(pooled), ownership_head->forward(h)};
}

mx::array TransformerMuZero::project(const mx::array& latent) {
  return projector->forward(latent);
}

mx::array TransformerMuZero::predict_projection(const mx::array& projection) {
  return predictor->forward(projection);
}

mx::array TransformerMuZero::reconstruct(const mx::array& latent) {
  return recon_head->forward(latent);
}

}  // namespace rlbg
