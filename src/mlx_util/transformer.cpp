#include <mlx_util/transformer.h>

#include <cmath>

#include <format>
#include <stdexcept>
#include <vector>

#include <mlx_util/activations.h>

namespace rlbg::nn {

namespace {

ArrayFn or_relu(ArrayFn activation) {
  return activation ? s::move(activation) : ArrayFn(nn::relu);
}

mx::array split_heads(const mx::array& x, int num_heads) {
  return mx::transpose(mx::unflatten(x, -1, {num_heads, -1}), {0, 2, 1, 3});
}

} // namespace

MultiHeadAttention::MultiHeadAttention(
  int dims,
  int num_heads,
  s::optional<int> query_input_dims,
  s::optional<int> key_input_dims,
  s::optional<int> value_input_dims,
  s::optional<int> value_dims,
  s::optional<int> value_output_dims,
  bool bias)
  : num_heads(num_heads)
{
  if (dims % num_heads != 0)
    throw s::invalid_argument(s::format(
      "The input feature dimensions should be divisible by the number of heads "
      "({} % {}) != 0", dims, num_heads
    ));
  int q_in = query_input_dims.value_or(dims);
  int k_in = key_input_dims.value_or(dims);
  int v_in = value_input_dims.value_or(k_in);
  int v_dims = value_dims.value_or(dims);
  int v_out = value_output_dims.value_or(dims);
  query_proj = register_module("query_proj", s::make_shared<Linear>(q_in, dims, bias));
  key_proj = register_module("key_proj", s::make_shared<Linear>(k_in, dims, bias));
  value_proj = register_module("value_proj", s::make_shared<Linear>(v_in, v_dims, bias));
  out_proj = register_module("out_proj", s::make_shared<Linear>(v_dims, v_out, bias));
}

mx::array MultiHeadAttention::forward(
  const mx::array& queries,
  const mx::array& keys,
  const mx::array& values,
  const Mask& mask)
{
  mx::array q = split_heads(query_proj->forward(queries), num_heads);
  mx::array k = split_heads(key_proj->forward(keys), num_heads);
  mx::array v = split_heads(value_proj->forward(values), num_heads);
  float scale = std::sqrt(1.0f / q.shape(-1));
  mx::array out = mx::fast::scaled_dot_product_attention(
    q, k, v, scale, mask ? "array" : "", mask
  );
  out = mx::flatten(mx::transpose(out, {0, 2, 1, 3}), -2, -1);
  return out_proj->forward(out);
}

mx::array MultiHeadAttention::create_additive_causal_mask(int n, mx::Dtype dtype) {
  mx::array indices = mx::arange(n);
  mx::array mask = mx::less(mx::expand_dims(indices, 1), mx::expand_dims(indices, 0));
  return mx::astype(mask, dtype) * mx::array(mx::finfo(dtype).min, dtype);
}

TransformerEncoderLayer::TransformerEncoderLayer(
  int dims,
  int num_heads,
  s::optional<int> mlp_dims,
  double dropout,
  ArrayFn activation,
  bool norm_first)
  : activation(or_relu(s::move(activation))), norm_first(norm_first)
{
  int mlp = mlp_dims.value_or(dims * 4);
  attention = register_module("attention", s::make_shared<MultiHeadAttention>(dims, num_heads));
  ln1 = register_module("ln1", s::make_shared<LayerNorm>(dims));
  ln2 = register_module("ln2", s::make_shared<LayerNorm>(dims));
  linear1 = register_module("linear1", s::make_shared<Linear>(dims, mlp));
  linear2 = register_module("linear2", s::make_shared<Linear>(mlp, dims));
  dropout1 = register_module("dropout1", s::make_shared<Dropout>(dropout));
  dropout2 = register_module("dropout2", s::make_shared<Dropout>(dropout));
}

mx::array TransformerEncoderLayer::forward(const mx::array& x, const Mask& mask) {
  if (norm_first) {
    mx::array y = ln1->forward(x);
    y = attention->forward(y, y, y, mask);
    y = dropout1->forward(y);
    mx::array h = x + y;
    y = ln2->forward(h);
    y = linear1->forward(y);
    y = activation(y);
    y = dropout2->forward(y);
    y = linear2->forward(y);
    return h + y;
  }
  mx::array y = attention->forward(x, x, x, mask);
  y = dropout1->forward(y);
  mx::array h = ln1->forward(x + y);
  y = linear1->forward(h);
  y = activation(y);
  y = dropout2->forward(y);
  y = linear2->forward(y);
  return ln2->forward(h + y);
}

TransformerEncoder::TransformerEncoder(
  int num_layers,
  int dims,
  int num_heads,
  s::optional<int> mlp_dims,
  double dropout,
  ArrayFn activation,
  bool norm_first,
  bool checkpoint)
  : layers(register_module("layers", s::make_shared<ModuleList>())),
    checkpoint(checkpoint)
{
  for (int i = 0; i < num_layers; ++i)
    layers->append(s::make_shared<TransformerEncoderLayer>(
      dims, num_heads, mlp_dims, dropout, activation, norm_first
    ));
  ln = register_module("ln", s::make_shared<LayerNorm>(dims));
}

mx::array TransformerEncoder::forward(const mx::array& x, const Mask& mask) {
  mx::array y = x;
  for (size_t i = 0; i < layers->size(); ++i) {
    auto layer = layers->at<TransformerEncoderLayer>(i);
    if (checkpoint) {
      s::vector<mx::array> inputs{y};
      if (mask) inputs.push_back(*mask);
      auto fn = [layer](const s::vector<mx::array>& xs) {
        Mask m = xs.size() > 1 ? Mask(xs[1]) : Mask();
        return s::vector<mx::array>{layer->forward(xs[0], m)};
      };
      y = nn::checkpoint(*layer, fn, inputs)[0];
    } else {
      y = layer->forward(y, mask);
    }
  }
  return ln->forward(y);
}

TransformerDecoderLayer::TransformerDecoderLayer(
  int dims,
  int num_heads,
  s::optional<int> mlp_dims,
  double dropout,
  ArrayFn activation,
  bool norm_first)
  : activation(or_relu(s::move(activation))), norm_first(norm_first)
{
  int mlp = mlp_dims.value_or(dims * 4);
  self_attention = register_module("self_attention", s::make_shared<MultiHeadAttention>(dims, num_heads));
  cross_attention = register_module("cross_attention", s::make_shared<MultiHeadAttention>(dims, num_heads));
  ln1 = register_module("ln1", s::make_shared<LayerNorm>(dims));
  ln2 = register_module("ln2", s::make_shared<LayerNorm>(dims));
  ln3 = register_module("ln3", s::make_shared<LayerNorm>(dims));
  linear1 = register_module("linear1", s::make_shared<Linear>(dims, mlp));
  linear2 = register_module("linear2", s::make_shared<Linear>(mlp, dims));
  dropout1 = register_module("dropout1", s::make_shared<Dropout>(dropout));
  dropout2 = register_module("dropout2", s::make_shared<Dropout>(dropout));
  dropout3 = register_module("dropout3", s::make_shared<Dropout>(dropout));
}

mx::array TransformerDecoderLayer::forward(
  const mx::array& x,
  const mx::array& memory,
  const Mask& x_mask,
  const Mask& memory_mask)
{
  if (norm_first) {
    mx::array y = ln1->forward(x);
    y = self_attention->forward(y, y, y, x_mask);
    y = dropout1->forward(y);
    mx::array h = x + y;
    y = ln2->forward(h);
    y = cross_attention->forward(y, memory, memory, memory_mask);
    y = dropout2->forward(y);
    h = h + y;
    y = ln3->forward(h);
    y = linear1->forward(y);
    y = activation(y);
    y = dropout3->forward(y);
    y = linear2->forward(y);
    return h + y;
  }
  mx::array y = self_attention->forward(x, x, x, x_mask);
  y = dropout1->forward(y);
  mx::array h = ln1->forward(x + y);
  y = cross_attention->forward(h, memory, memory, memory_mask);
  y = dropout2->forward(y);
  h = ln2->forward(h + y);
  y = linear1->forward(h);
  y = activation(y);
  y = dropout3->forward(y);
  y = linear2->forward(y);
  return ln3->forward(h + y);
}

TransformerDecoder::TransformerDecoder(
  int num_layers,
  int dims,
  int num_heads,
  s::optional<int> mlp_dims,
  double dropout,
  ArrayFn activation,
  bool norm_first,
  bool checkpoint)
  : layers(register_module("layers", s::make_shared<ModuleList>())),
    checkpoint(checkpoint)
{
  for (int i = 0; i < num_layers; ++i)
    layers->append(s::make_shared<TransformerDecoderLayer>(
      dims, num_heads, mlp_dims, dropout, activation, norm_first
    ));
  ln = register_module("ln", s::make_shared<LayerNorm>(dims));
}

mx::array TransformerDecoder::forward(
  const mx::array& x,
  const mx::array& memory,
  const Mask& x_mask,
  const Mask& memory_mask)
{
  mx::array y = x;
  for (size_t i = 0; i < layers->size(); ++i) {
    auto layer = layers->at<TransformerDecoderLayer>(i);
    if (checkpoint) {
      // memory is an input so its gradient flows back to the encoder
      s::vector<mx::array> inputs{y, memory};
      if (x_mask) inputs.push_back(*x_mask);
      if (memory_mask) inputs.push_back(*memory_mask);
      bool has_x_mask = x_mask.has_value();
      bool has_memory_mask = memory_mask.has_value();
      auto fn = [layer, has_x_mask, has_memory_mask](const s::vector<mx::array>& xs) {
        size_t at = 2;
        Mask xm = has_x_mask ? Mask(xs[at++]) : Mask();
        Mask mm = has_memory_mask ? Mask(xs[at++]) : Mask();
        return s::vector<mx::array>{layer->forward(xs[0], xs[1], xm, mm)};
      };
      y = nn::checkpoint(*layer, fn, inputs)[0];
    } else {
      y = layer->forward(y, memory, x_mask, memory_mask);
    }
  }
  return ln->forward(y);
}

Transformer::Transformer(
  int dims,
  int num_heads,
  int num_encoder_layers,
  int num_decoder_layers,
  s::optional<int> mlp_dims,
  double dropout,
  ArrayFn activation,
  s::shared_ptr<TransformerEncoder> custom_encoder,
  s::shared_ptr<TransformerDecoder> custom_decoder,
  bool norm_first,
  bool checkpoint)
{
  encoder = register_module(
    "encoder",
    custom_encoder ? custom_encoder
                   : s::make_shared<TransformerEncoder>(
                       num_encoder_layers, dims, num_heads, mlp_dims, dropout,
                       activation, norm_first, checkpoint
                     )
  );
  decoder = register_module(
    "decoder",
    custom_decoder ? custom_decoder
                   : s::make_shared<TransformerDecoder>(
                       num_decoder_layers, dims, num_heads, mlp_dims, dropout,
                       activation, norm_first, checkpoint
                     )
  );
}

mx::array Transformer::forward(
  const mx::array& src,
  const mx::array& tgt,
  const Mask& src_mask,
  const Mask& tgt_mask,
  const Mask& memory_mask)
{
  mx::array memory = encoder->forward(src, src_mask);
  return decoder->forward(tgt, memory, tgt_mask, memory_mask);
}

} // namespace rlbg::nn
