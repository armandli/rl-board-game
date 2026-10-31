#ifndef RLBG_MLX_UTIL_TRANSFORMER_H
#define RLBG_MLX_UTIL_TRANSFORMER_H

#include <memory>
#include <optional>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/containers.h>
#include <mlx_util/dropout.h>
#include <mlx_util/linear.h>
#include <mlx_util/module.h>
#include <mlx_util/normalization.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

using Mask = s::optional<mx::array>;

// unset dims follow Python's defaults: query/key inputs and value dims are
// dims, value inputs are key_input_dims
struct MultiHeadAttention : Module {
  MultiHeadAttention(
    int dims,
    int num_heads,
    s::optional<int> query_input_dims = s::nullopt,
    s::optional<int> key_input_dims = s::nullopt,
    s::optional<int> value_input_dims = s::nullopt,
    s::optional<int> value_dims = s::nullopt,
    s::optional<int> value_output_dims = s::nullopt,
    bool bias = false
  );

  s::string type_name() const override { return "MultiHeadAttention"; }
  // mask is additive or boolean and broadcasts to (N, heads, L_q, L_k)
  mx::array forward(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    const Mask& mask = s::nullopt
  );
  mx::array operator()(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    const Mask& mask = s::nullopt)
  {
    return forward(queries, keys, values, mask);
  }

  static mx::array create_additive_causal_mask(int n, mx::Dtype dtype = mx::float32);

  int num_heads;
  s::shared_ptr<Linear> query_proj;
  s::shared_ptr<Linear> key_proj;
  s::shared_ptr<Linear> value_proj;
  s::shared_ptr<Linear> out_proj;
};

struct TransformerEncoderLayer : Module {
  TransformerEncoderLayer(
    int dims,
    int num_heads,
    s::optional<int> mlp_dims = s::nullopt,
    double dropout = 0.0,
    ArrayFn activation = nullptr,
    bool norm_first = true
  );

  s::string type_name() const override { return "TransformerEncoderLayer"; }
  mx::array forward(const mx::array& x, const Mask& mask);
  mx::array operator()(const mx::array& x, const Mask& mask) { return forward(x, mask); }

  s::shared_ptr<MultiHeadAttention> attention;
  s::shared_ptr<LayerNorm> ln1;
  s::shared_ptr<LayerNorm> ln2;
  s::shared_ptr<Linear> linear1;
  s::shared_ptr<Linear> linear2;
  s::shared_ptr<Dropout> dropout1;
  s::shared_ptr<Dropout> dropout2;
  ArrayFn activation;
  bool norm_first;
};

// checkpoint recomputes each layer in the backward pass to save memory
struct TransformerEncoder : Module {
  TransformerEncoder(
    int num_layers,
    int dims,
    int num_heads,
    s::optional<int> mlp_dims = s::nullopt,
    double dropout = 0.0,
    ArrayFn activation = nullptr,
    bool norm_first = true,
    bool checkpoint = false
  );

  s::string type_name() const override { return "TransformerEncoder"; }
  virtual mx::array forward(const mx::array& x, const Mask& mask);
  mx::array operator()(const mx::array& x, const Mask& mask) { return forward(x, mask); }

  s::shared_ptr<ModuleList> layers;
  s::shared_ptr<LayerNorm> ln;
  bool checkpoint;
};

struct TransformerDecoderLayer : Module {
  TransformerDecoderLayer(
    int dims,
    int num_heads,
    s::optional<int> mlp_dims = s::nullopt,
    double dropout = 0.0,
    ArrayFn activation = nullptr,
    bool norm_first = true
  );

  s::string type_name() const override { return "TransformerDecoderLayer"; }
  mx::array forward(
    const mx::array& x,
    const mx::array& memory,
    const Mask& x_mask,
    const Mask& memory_mask
  );
  mx::array operator()(
    const mx::array& x,
    const mx::array& memory,
    const Mask& x_mask,
    const Mask& memory_mask)
  {
    return forward(x, memory, x_mask, memory_mask);
  }

  s::shared_ptr<MultiHeadAttention> self_attention;
  s::shared_ptr<MultiHeadAttention> cross_attention;
  s::shared_ptr<LayerNorm> ln1;
  s::shared_ptr<LayerNorm> ln2;
  s::shared_ptr<LayerNorm> ln3;
  s::shared_ptr<Linear> linear1;
  s::shared_ptr<Linear> linear2;
  s::shared_ptr<Dropout> dropout1;
  s::shared_ptr<Dropout> dropout2;
  s::shared_ptr<Dropout> dropout3;
  ArrayFn activation;
  bool norm_first;
};

struct TransformerDecoder : Module {
  TransformerDecoder(
    int num_layers,
    int dims,
    int num_heads,
    s::optional<int> mlp_dims = s::nullopt,
    double dropout = 0.0,
    ArrayFn activation = nullptr,
    bool norm_first = true,
    bool checkpoint = false
  );

  s::string type_name() const override { return "TransformerDecoder"; }
  virtual mx::array forward(
    const mx::array& x,
    const mx::array& memory,
    const Mask& x_mask,
    const Mask& memory_mask
  );
  mx::array operator()(
    const mx::array& x,
    const mx::array& memory,
    const Mask& x_mask,
    const Mask& memory_mask)
  {
    return forward(x, memory, x_mask, memory_mask);
  }

  s::shared_ptr<ModuleList> layers;
  s::shared_ptr<LayerNorm> ln;
  bool checkpoint;
};

// custom_encoder / custom_decoder replace the default stacks; derive from
// TransformerEncoder / TransformerDecoder and override forward
struct Transformer : Module {
  explicit Transformer(
    int dims = 512,
    int num_heads = 8,
    int num_encoder_layers = 6,
    int num_decoder_layers = 6,
    s::optional<int> mlp_dims = s::nullopt,
    double dropout = 0.0,
    ArrayFn activation = nullptr,
    s::shared_ptr<TransformerEncoder> custom_encoder = nullptr,
    s::shared_ptr<TransformerDecoder> custom_decoder = nullptr,
    bool norm_first = true,
    bool checkpoint = false
  );

  s::string type_name() const override { return "Transformer"; }
  mx::array forward(
    const mx::array& src,
    const mx::array& tgt,
    const Mask& src_mask,
    const Mask& tgt_mask,
    const Mask& memory_mask
  );
  mx::array operator()(
    const mx::array& src,
    const mx::array& tgt,
    const Mask& src_mask,
    const Mask& tgt_mask,
    const Mask& memory_mask)
  {
    return forward(src, tgt, src_mask, tgt_mask, memory_mask);
  }

  s::shared_ptr<TransformerEncoder> encoder;
  s::shared_ptr<TransformerDecoder> decoder;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_TRANSFORMER_H
