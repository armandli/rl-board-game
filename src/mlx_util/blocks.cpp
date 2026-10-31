#include <mlx_util/blocks.h>

#include <format>

#include <mlx_util/activations.h>

namespace rlbg::nn {

SwiGLU::SwiGLU(int dims, int hidden_dims, bool bias)
  : w1(register_module("w1", s::make_shared<Linear>(dims, hidden_dims, bias))),
    w2(register_module("w2", s::make_shared<Linear>(hidden_dims, dims, bias))),
    w3(register_module("w3", s::make_shared<Linear>(dims, hidden_dims, bias)))
{}

s::string SwiGLU::extra_repr() const {
  return s::format("dims={}, hidden_dims={}", w1->weight.shape(1), w1->weight.shape(0));
}

mx::array SwiGLU::forward(const mx::array& x) {
  return w2->forward(nn::silu(w1->forward(x)) * w3->forward(x));
}

int swiglu_hidden_dims(int dims) {
  int hidden = (8 * dims + 2) / 3;
  return (hidden + 31) / 32 * 32;
}

TransformerBlock::TransformerBlock(int dims, int num_heads, s::optional<int> mlp_dims)
  : attention_norm(register_module("attention_norm", s::make_shared<RMSNorm>(dims))),
    attention(register_module("attention", s::make_shared<MultiHeadAttention>(dims, num_heads))),
    mlp_norm(register_module("mlp_norm", s::make_shared<RMSNorm>(dims))),
    mlp(register_module("mlp", s::make_shared<SwiGLU>(dims, mlp_dims.value_or(swiglu_hidden_dims(dims)))))
{}

mx::array TransformerBlock::forward(const mx::array& x, const Mask& mask) {
  mx::array y = attention_norm->forward(x);
  mx::array h = x + attention->forward(y, y, y, mask);
  return h + mlp->forward(mlp_norm->forward(h));
}

} // namespace rlbg::nn
