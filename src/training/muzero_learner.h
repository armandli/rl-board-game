#ifndef RLBG_TRAINING_MUZERO_LEARNER_H
#define RLBG_TRAINING_MUZERO_LEARNER_H

#include <cstdint>

#include <memory>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include <model/muzero_model.h>
#include <training/batch.h>
#include <training/learner.h>

namespace rlbg {

namespace s = std;
namespace mx = mlx::core;

struct MuZeroLearnerConfig {
  // "muon": Muon on the transformer blocks' weight matrices, AdamW on the
  // rest; "adamw": AdamW everywhere
  s::string optimizer = "muon";
  double    learning_rate = 1e-3;     // AdamW
  double    muon_learning_rate = 0.02;
  double    weight_decay = 0.01;
  int       warmup_steps = 1000;
  int       total_steps = 300000;     // cosine decay horizon
  double    final_lr_fraction = 0.05;
  double    grad_clip = 1.0;
  // precision of the forward and backward pass; weights and optimizer
  // state stay float32
  s::string compute_dtype = "bfloat16";
  double    dynamics_gradient_scale = 0.5;

  double    policy_weight = 1.0;
  double    value_weight = 1.0;
  double    reward_weight = 1.0;
  double    consistency_weight = 2.0;
  double    reconstruction_weight = 0.5;
  double    ownership_weight = 0.5;
};

// MuZero training on K step unrolls:
//   policy: cross entropy to the search policy
//   value, reward: cross entropy over loss/draw/win
//   consistency: EfficientZero's SimSiam loss, negative cosine between the
//     predicted projection of the unrolled latent and the stop-gradient
//     projection of the representation of the real state t+k
//   reconstruction: decode the real state's token classes from the latent
//     (Scholz et al. 2021), which grounds the dynamics in the board
//   ownership: final owner of every square, KataGo's auxiliary target that
//     splits the win/loss signal into per-square credit
// Terms at k >= 1 are scaled by 1/K and the latent's gradient is halved
// entering each dynamics step, as in MuZero.
struct MuZeroLearner : Learner {
  MuZeroLearner(s::shared_ptr<MuZeroModel> model, const MuZeroLearnerConfig& config);

  LearnerStats train_step(const TrainingBatch& batch) override;
  int64_t step() const override { return mStep; }
  nn::Params weights() const override;
  MuZeroModel& model() override { return *mModel; }

  void save_checkpoint(const s::string& dir) const override;
  bool load_checkpoint(const s::string& dir) override;

protected:
  optim::Schedule schedule(double peak) const;

  s::shared_ptr<MuZeroModel>       mModel;
  MuZeroLearnerConfig              mConfig;
  mx::Dtype                        mCompute;
  s::shared_ptr<optim::Optimizer>  mOptimizer;
  int64_t                          mStep = 0;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_MUZERO_LEARNER_H
