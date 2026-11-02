#ifndef RLBG_LIBTORCH_LEARNER_H
#define RLBG_LIBTORCH_LEARNER_H

#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <training/batch.h>
#include <training/learner_stats.h>

#include <libtorch/device.h>
#include <libtorch/transformer_muzero.h>

namespace rlbg::libtorch {

namespace s = std;
namespace t = torch;

struct LearnerConfig {
  // "muon": Muon on the transformer blocks' weight matrices, AdamW on the
  // rest; "adamw": AdamW everywhere
  s::string optimizer = "muon";
  double    learning_rate = 1e-3;     // AdamW peak
  double    muon_learning_rate = 0.02;
  double    weight_decay = 0.01;
  int       warmup_steps = 1000;
  int       total_steps = 100000;     // cosine decay horizon
  double    final_lr_fraction = 0.05;
  double    grad_clip = 1.0;
  // bfloat16 autocast of the forward and backward pass on CUDA; weights and
  // optimizer state stay float32
  bool      bf16 = true;
  double    dynamics_gradient_scale = 0.5;

  double    policy_weight = 1.0;
  double    value_weight = 1.0;
  double    reward_weight = 1.0;
  double    consistency_weight = 2.0;
  double    reconstruction_weight = 0.5;
  double    ownership_weight = 0.5;
};

// Muon (Jordan et al. 2024): momentum orthogonalized by a quintic
// Newton-Schulz iteration, for 2D hidden weight matrices; matches
// mlx.optimizers.Muon (coupled weight decay, Nesterov momentum, learning rate
// scaled by sqrt(max(1, rows / cols)))
struct Muon {
  Muon(s::vector<t::Tensor> params, double momentum = 0.95, double weight_decay = 0.01, int ns_steps = 5);
  void step(double lr);

  void save(t::serialize::OutputArchive& archive) const;
  void load(t::serialize::InputArchive& archive);

  s::vector<t::Tensor> params;
  s::vector<t::Tensor> momentum_buffers;
  double               momentum;
  double               weight_decay;
  int                  ns_steps;
};

// MuZero training on K step unrolls:
//   policy: cross entropy to the search policy
//   value, reward: cross entropy over loss/draw/win; value targets may be
//     soft (bootstrapped, see ValueTargetConfig)
//   consistency: EfficientZero's SimSiam loss, negative cosine between the
//     predicted projection of the unrolled latent and the stop-gradient
//     projection of the representation of the real state t+k
//   reconstruction: decode the real state's token classes from the latent
//     (Scholz et al. 2021), which grounds the dynamics in the board
//   ownership: final owner of every square, KataGo's auxiliary target that
//     splits the win/loss signal into per-square credit
// Terms at k >= 1 are scaled by 1/K and the latent's gradient is scaled by
// dynamics_gradient_scale entering each dynamics step, as in MuZero. Linear
// warmup then cosine decay of the learning rates. Runs on its own CUDA stream.
struct Learner {
  Learner(const TransformerMuZeroConfig& model, const LearnerConfig& config, t::Device device);

  LearnerStats train_step(const TrainingBatch& batch);
  int64_t step() const { return mStep; }
  TransformerMuZero& model() { return mModel; }
  const DeviceStream& stream() const { return mStream; }
  // learning rate multiplier of a step: warmup, then cosine decay
  double schedule(int64_t step) const;

  // model.pt, optimizer.pt and learner.txt in dir, written atomically
  void save_checkpoint(const s::string& dir);
  // false when dir holds no checkpoint
  bool load_checkpoint(const s::string& dir);
  void save_weights(const s::string& file);
  // a copy of the current weights in dtype, complete when this returns
  TransformerMuZero copy_model(t::ScalarType dtype);

protected:
  TransformerMuZero              mModel;
  LearnerConfig                  mConfig;
  t::Device                      mDevice;
  DeviceStream                   mStream;
  s::unique_ptr<t::optim::AdamW> mAdam;
  s::unique_ptr<Muon>            mMuon;
  int64_t                        mStep = 0;
};

}  // namespace rlbg::libtorch

#endif//RLBG_LIBTORCH_LEARNER_H
