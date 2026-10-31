#ifndef RLBG_MLX_UTIL_ACTIVATIONS_H
#define RLBG_MLX_UTIL_ACTIVATIONS_H

#include <string>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// call these qualified (nn::sigmoid) or ADL also finds mx::sigmoid
mx::array sigmoid(const mx::array& x);
mx::array relu(const mx::array& x);
mx::array relu2(const mx::array& x);
mx::array relu6(const mx::array& x);
mx::array leaky_relu(const mx::array& x, double negative_slope = 0.01);
mx::array log_softmax(const mx::array& x, int axis = -1);
mx::array elu(const mx::array& x, double alpha = 1.0);
mx::array softmax(const mx::array& x, int axis = -1);
mx::array softplus(const mx::array& x);
mx::array softsign(const mx::array& x);
mx::array softshrink(const mx::array& x, double lambd = 0.5);
mx::array celu(const mx::array& x, double alpha = 1.0);
mx::array silu(const mx::array& x);
mx::array log_sigmoid(const mx::array& x);
mx::array gelu(const mx::array& x);
mx::array gelu_approx(const mx::array& x);
mx::array gelu_fast_approx(const mx::array& x);
mx::array glu(const mx::array& x, int axis = -1);
mx::array step(const mx::array& x, double threshold = 0.0);
mx::array selu(const mx::array& x);
mx::array prelu(const mx::array& x, const mx::array& alpha);
mx::array mish(const mx::array& x);
mx::array hardswish(const mx::array& x);
mx::array hard_tanh(const mx::array& x, double min_val = -1.0, double max_val = 1.0);
mx::array hard_shrink(const mx::array& x, double lambd = 0.5);
mx::array softmin(const mx::array& x, int axis = -1);
mx::array tanh(const mx::array& x);

#define RLBG_NN_ACTIVATION(NAME, FN)                                  \
  struct NAME : UnaryModule {                                         \
    s::string type_name() const override { return #NAME; }            \
    mx::array forward(const mx::array& x) override { return FN(x); }  \
  };

RLBG_NN_ACTIVATION(Sigmoid, nn::sigmoid)
RLBG_NN_ACTIVATION(Mish, nn::mish)
RLBG_NN_ACTIVATION(ReLU, nn::relu)
RLBG_NN_ACTIVATION(ReLU2, nn::relu2)
RLBG_NN_ACTIVATION(ReLU6, nn::relu6)
RLBG_NN_ACTIVATION(Softmax, nn::softmax)
RLBG_NN_ACTIVATION(Softplus, nn::softplus)
RLBG_NN_ACTIVATION(Softsign, nn::softsign)
RLBG_NN_ACTIVATION(SiLU, nn::silu)
RLBG_NN_ACTIVATION(LogSoftmax, nn::log_softmax)
RLBG_NN_ACTIVATION(LogSigmoid, nn::log_sigmoid)
RLBG_NN_ACTIVATION(Tanh, nn::tanh)
RLBG_NN_ACTIVATION(Hardswish, nn::hardswish)
RLBG_NN_ACTIVATION(SELU, nn::selu)
RLBG_NN_ACTIVATION(HardTanh, nn::hard_tanh)
RLBG_NN_ACTIVATION(Softmin, nn::softmin)

#undef RLBG_NN_ACTIVATION

struct GLU : UnaryModule {
  explicit GLU(int axis = -1) : axis(axis) {}
  s::string type_name() const override { return "GLU"; }
  mx::array forward(const mx::array& x) override { return nn::glu(x, axis); }

  int axis;
};

struct LeakyReLU : UnaryModule {
  explicit LeakyReLU(double negative_slope = 1e-2) : negative_slope(negative_slope) {}
  s::string type_name() const override { return "LeakyReLU"; }
  mx::array forward(const mx::array& x) override {
    return nn::leaky_relu(x, negative_slope);
  }

  double negative_slope;
};

struct ELU : UnaryModule {
  explicit ELU(double alpha = 1.0) : alpha(alpha) {}
  s::string type_name() const override { return "ELU"; }
  mx::array forward(const mx::array& x) override { return nn::elu(x, alpha); }

  double alpha;
};

struct CELU : UnaryModule {
  explicit CELU(double alpha = 1.0) : alpha(alpha) {}
  s::string type_name() const override { return "CELU"; }
  mx::array forward(const mx::array& x) override { return nn::celu(x, alpha); }

  double alpha;
};

struct Softshrink : UnaryModule {
  explicit Softshrink(double lambd = 0.5) : lambd(lambd) {}
  s::string type_name() const override { return "Softshrink"; }
  mx::array forward(const mx::array& x) override {
    return nn::softshrink(x, lambd);
  }

  double lambd;
};

struct HardShrink : UnaryModule {
  explicit HardShrink(double lambd = 0.5) : lambd(lambd) {}
  s::string type_name() const override { return "HardShrink"; }
  mx::array forward(const mx::array& x) override {
    return nn::hard_shrink(x, lambd);
  }

  double lambd;
};

struct Step : UnaryModule {
  explicit Step(double threshold = 0.0) : threshold(threshold) {}
  s::string type_name() const override { return "Step"; }
  mx::array forward(const mx::array& x) override { return nn::step(x, threshold); }

  double threshold;
};

struct PReLU : UnaryModule {
  explicit PReLU(int num_parameters = 1, float init = 0.25f)
    : weight(mx::full({num_parameters}, init)) {
    register_parameter("weight", weight);
  }
  s::string type_name() const override { return "PReLU"; }
  mx::array forward(const mx::array& x) override { return nn::prelu(x, weight); }

  mx::array weight;
};

// approx is "none", "precise", "tanh" or "fast"
struct GELU : UnaryModule {
  explicit GELU(s::string approx = "none");
  s::string type_name() const override { return "GELU"; }
  mx::array forward(const mx::array& x) override;

  s::string approx;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_ACTIVATIONS_H
