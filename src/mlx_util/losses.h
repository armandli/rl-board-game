#ifndef RLBG_MLX_UTIL_LOSSES_H
#define RLBG_MLX_UTIL_LOSSES_H

#include <optional>

#include <mlx/mlx.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

enum struct Reduction : int {
  None,
  Mean,
  Sum,
};

// mlx.nn.losses; default reductions match Python's for each loss

// targets are class indices, or probabilities with the same shape as logits
mx::array cross_entropy(
  const mx::array& logits,
  const mx::array& targets,
  const s::optional<mx::array>& weights = s::nullopt,
  int axis = -1,
  double label_smoothing = 0.0,
  Reduction reduction = Reduction::None
);
mx::array binary_cross_entropy(
  const mx::array& inputs,
  const mx::array& targets,
  const s::optional<mx::array>& weights = s::nullopt,
  bool with_logits = true,
  Reduction reduction = Reduction::Mean
);
mx::array l1_loss(
  const mx::array& predictions,
  const mx::array& targets,
  Reduction reduction = Reduction::Mean
);
mx::array mse_loss(
  const mx::array& predictions,
  const mx::array& targets,
  Reduction reduction = Reduction::Mean
);
mx::array nll_loss(
  const mx::array& inputs,
  const mx::array& targets,
  int axis = -1,
  Reduction reduction = Reduction::None
);
mx::array gaussian_nll_loss(
  const mx::array& inputs,
  const mx::array& targets,
  const mx::array& vars,
  bool full = false,
  double eps = 1e-6,
  Reduction reduction = Reduction::Mean
);
mx::array kl_div_loss(
  const mx::array& inputs,
  const mx::array& targets,
  int axis = -1,
  Reduction reduction = Reduction::None
);
mx::array smooth_l1_loss(
  const mx::array& predictions,
  const mx::array& targets,
  double beta = 1.0,
  Reduction reduction = Reduction::Mean
);
mx::array triplet_loss(
  const mx::array& anchors,
  const mx::array& positives,
  const mx::array& negatives,
  int axis = -1,
  int p = 2,
  double margin = 1.0,
  double eps = 1e-6,
  Reduction reduction = Reduction::None
);
mx::array hinge_loss(
  const mx::array& inputs,
  const mx::array& targets,
  Reduction reduction = Reduction::None
);
mx::array huber_loss(
  const mx::array& inputs,
  const mx::array& targets,
  double delta = 1.0,
  Reduction reduction = Reduction::None
);
mx::array log_cosh_loss(
  const mx::array& inputs,
  const mx::array& targets,
  Reduction reduction = Reduction::None
);
mx::array cosine_similarity_loss(
  const mx::array& x1,
  const mx::array& x2,
  int axis = 1,
  double eps = 1e-8,
  Reduction reduction = Reduction::None
);
mx::array margin_ranking_loss(
  const mx::array& inputs1,
  const mx::array& inputs2,
  const mx::array& targets,
  double margin = 0.0,
  Reduction reduction = Reduction::None
);

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_LOSSES_H
