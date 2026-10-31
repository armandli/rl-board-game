#ifndef RLBG_MLX_UTIL_TEST_ZOO_H
#define RLBG_MLX_UTIL_TEST_ZOO_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

// models and optimizers mirrored one for one by python_compat.py; any
// change here must be made there too

namespace rlbg::test {

namespace s = std;
namespace mx = mlx::core;

using nn::Params;

struct ZooEntry {
  s::string name;
  // file extension the producer writes this model's weights with
  s::string py_format;
  s::string cpp_format;
  s::function<nn::ModulePtr()> build;
  // run before the producer saves weights; updates BatchNorm statistics
  s::function<void(nn::Module&, const Params&)> warmup;
  // outputs keyed "<name>.<output>"
  s::function<Params(nn::Module&, const Params&)> run;
};

const s::vector<ZooEntry>& zoo();

// outputs of stateless functions: activations, losses, upsampling, ALiBi
Params functions_outputs(const Params& inputs);

nn::ModulePtr build_train_model();
mx::array train_loss(nn::Module& model, const Params& inputs);

struct OptEntry {
  s::string name;
  s::function<s::shared_ptr<optim::Optimizer>()> build;
  // clip gradients to this norm before updating, when positive
  double clip = 0.0;
};

const s::vector<OptEntry>& optimizers();

// one optimizer step on the training batch
void train_step(nn::Module& model, optim::Optimizer& opt, const OptEntry& entry, const Params& inputs);

} // namespace rlbg::test

#endif//RLBG_MLX_UTIL_TEST_ZOO_H
