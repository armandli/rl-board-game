#ifndef RLBG_MODEL_MUZERO_MODEL_H
#define RLBG_MODEL_MUZERO_MODEL_H

#include <string>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

namespace rlbg {

namespace s = std;
namespace mx = mlx::core;

// value and reward heads predict 3 outcome classes from the mover's point of
// view (loss, draw, win, i.e. -1, 0, +1); the scalar is P(win) - P(loss)
constexpr int NUM_OUTCOME_CLASSES = 3;

inline mx::array outcome_scalar(const mx::array& logits) {
  mx::array p = mx::softmax(logits, -1, true);
  return mx::take(p, 2, -1) - mx::take(p, 0, -1);
}

// -1, 0, +1 to class 0, 1, 2
inline int outcome_class(int outcome) {
  return outcome + 1;
}

struct Prediction {
  mx::array policy_logits;    // (B, A)
  mx::array value_logits;     // (B, 3)
  mx::array ownership_logits; // (B, T, C) token classes at the end of the game
};

struct DynamicsOutput {
  mx::array latent;           // (B, T, D)
  mx::array reward_logits;    // (B, 3) reward of the action for its mover
};

// MuZero's three learned functions plus the auxiliary heads used by the
// self-supervised losses:
//   represent  h: observation -> latent state s0
//   dynamics   g: (s, a) -> (s', r)
//   predict    f: s -> (policy, value)
//   project / predict_projection: SimSiam consistency heads (EfficientZero)
//   reconstruct: token classes of the observation the latent stands for
// Latents are (B, T, D) sequences of token embeddings. Arrays are lazy; the
// caller decides what to evaluate, so unused heads cost nothing.
struct MuZeroModel : nn::Module {
  s::string type_name() const override { return "MuZeroModel"; }

  // features (B, T, F) float
  virtual mx::array represent(const mx::array& features) = 0;
  // actions (B,) int32
  virtual DynamicsOutput dynamics(const mx::array& latent, const mx::array& actions) = 0;
  virtual Prediction predict(const mx::array& latent) = 0;
  virtual mx::array project(const mx::array& latent) = 0;
  virtual mx::array predict_projection(const mx::array& projection) = 0;
  // (B, T, C) logits
  virtual mx::array reconstruct(const mx::array& latent) = 0;

  virtual int num_tokens() const = 0;
  virtual int latent_dims() const = 0;
  int latent_size() const { return num_tokens() * latent_dims(); }
  virtual int num_actions() const = 0;

  size_t num_parameters() const {
    size_t n = 0;
    for (const auto& [k, v] : parameters()) n += v.size();
    return n;
  }
};

}  // namespace rlbg

#endif//RLBG_MODEL_MUZERO_MODEL_H
