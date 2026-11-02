#ifndef RLBG_EVALUATION_SNAPSHOT_EVALUATOR_H
#define RLBG_EVALUATION_SNAPSHOT_EVALUATOR_H

#include <cstdint>

#include <functional>
#include <memory>
#include <string>

#include <mlx/mlx.h>

#include <muzero_game.h>

#include <evaluation/basic_snapshot_evaluator.h>
#include <model/model_inference.h>
#include <model/muzero_model.h>

namespace rlbg {

namespace s = std;
namespace mx = mlx::core;

// the snapshot evaluator over MLX models saved as .safetensors
template <MuZeroGame Game>
struct SnapshotEvaluator : BasicSnapshotEvaluator<Game> {
  using ModelFactory = s::function<s::shared_ptr<MuZeroModel>()>;
  using SearchMaker = typename BasicSnapshotEvaluator<Game>::SearchMaker;

  SnapshotEvaluator(
      const EvaluatorConfig& config,
      const s::string& dir,
      ModelFactory make_model,
      int observation_size,
      mx::Dtype dtype,
      SearchMaker make_search,
      uint64_t seed)
  : BasicSnapshotEvaluator<Game>(
      config, dir,
      [make_model, observation_size, dtype]{
        return s::shared_ptr<SnapshotInference>(s::make_shared<ModelInference>(make_model(), observation_size, dtype));
      },
      s::move(make_search), seed)
  {}
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_SNAPSHOT_EVALUATOR_H
