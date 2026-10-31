#ifndef RLBG_MLX_UTIL_EMBEDDING_H
#define RLBG_MLX_UTIL_EMBEDDING_H

#include <string>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

struct Embedding : UnaryModule {
  Embedding(int num_embeddings, int dims);

  s::string type_name() const override { return "Embedding"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x) override { return mx::take(weight, x, 0); }
  // uses the embedding matrix as a linear layer, e.g. for tied output heads
  mx::array as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(weight));
  }

  mx::array weight;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_EMBEDDING_H
