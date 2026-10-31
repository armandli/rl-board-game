#include <mlx_util/embedding.h>

#include <cmath>

#include <format>

namespace rlbg::nn {

Embedding::Embedding(int num_embeddings, int dims)
  : weight(mx::random::normal(
      {num_embeddings, dims},
      mx::float32,
      0.0f,
      std::sqrt(1.0f / dims)
    ))
{
  register_parameter("weight", weight);
}

s::string Embedding::extra_repr() const {
  return s::format("{}, {}", weight.shape(0), weight.shape(1));
}

} // namespace rlbg::nn
