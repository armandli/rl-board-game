# mlx_util

C++ counterpart of Python's `mlx.nn` and `mlx.optimizers`, built on the MLX C++
core (`mlx::core`), which has no module, layer or optimizer classes. Include
`<mlx_util/nn.h>` and link the `mlx_util` CMake target.

| Python                         | C++                                             |
|--------------------------------|-------------------------------------------------|
| `mlx.nn.Module`                | `rlbg::nn::Module` (`UnaryModule` for `x -> y`) |
| `mlx.nn.<Layer>`               | `rlbg::nn::<Layer>`, same names and arguments   |
| `mlx.nn.value_and_grad`        | `rlbg::nn::value_and_grad`                      |
| `mlx.nn.losses.*`, `mlx.nn.init.*` | `rlbg::nn::*`, `rlbg::nn::init::*`          |
| `mlx.optimizers.*`             | `rlbg::optim::*`                                |
| `mx.load` / `mx.savez` / `mx.save_safetensors` | `nn::load_arrays` / `nn::save_npz` / `nn::save_safetensors` |

Parameters are flattened to `nn::Params` (`std::map<std::string, mx::array>`)
keyed exactly like `tree_flatten(model.parameters())` in Python, e.g.
`layers.0.weight`.

## Writing a model

```cpp
#include <mlx_util/nn.h>

namespace nn = rlbg::nn;
namespace optim = rlbg::optim;
namespace mx = mlx::core;

struct PolicyValue : nn::Module {
  PolicyValue()
    : trunk(register_module("trunk", std::make_shared<nn::Sequential>(
        std::make_shared<nn::Conv2d>(2, 32, 3, 1, 1),
        std::make_shared<nn::BatchNorm>(32),
        nn::relu))),
      policy(register_module("policy", std::make_shared<nn::Linear>(32 * 42, 7))),
      value(register_module("value", std::make_shared<nn::Linear>(32 * 42, 1))) {}

  std::pair<mx::array, mx::array> forward(const mx::array& x) {
    mx::array h = mx::flatten((*trunk)(x), 1);
    return {policy->forward(h), mx::tanh(value->forward(h))};
  }

  std::shared_ptr<nn::Sequential> trunk;
  std::shared_ptr<nn::Linear> policy;
  std::shared_ptr<nn::Linear> value;
};

auto model = std::make_shared<PolicyValue>();
optim::AdamW opt(optim::cosine_decay(1e-3, 10000));
auto loss_and_grad = nn::value_and_grad(*model, [&](const mx::array& x, const mx::array& pi, const mx::array& z) {
  auto [logits, v] = model->forward(x);
  return nn::cross_entropy(logits, pi, std::nullopt, -1, 0.0, nn::Reduction::Mean) + nn::mse_loss(v, z);
});
auto [loss, grads] = loss_and_grad(x, pi, z);
opt.update(*model, grads);
nn::eval(model->parameters());

model->save_weights("model.safetensors");   // or .npz
opt.save_state("optimizer.safetensors");
```

Register every parameter (`register_parameter`) and child (`register_module`)
under the attribute name the equivalent Python class uses; those names are the
file keys. Modules hold their parameters by reference, so they are neither
copyable nor movable: create them with `std::make_shared`.

## Sharing files with Python

The equivalent Python model reads and writes the same files:

```python
model.load_weights("model.safetensors")      # strict: same keys and shapes
optimizer.state = tree_unflatten(list(mx.load("optimizer.safetensors").items()))
# ... and back
model.save_weights("model.npz")
mx.save_safetensors("optimizer.safetensors", dict(tree_flatten(optimizer.state)))
```

`.npz` (also compressed, when built with zlib) and `.safetensors` work both
ways; `.gguf` can be loaded.

## Differences from Python

- Not ported: quantized layers (`QuantizedLinear`, ...) and distributed layers
  and helpers (`average_gradients`, sharded linears).
- `Module::update_modules` is not provided; layers keep typed pointers to
  their children.
- Glorot/He initializers take `gain`/`mode` when created, not when called.
- `value_and_grad` puts the model's untraced parameters back after the call.
- `BatchNorm` keeps its running statistics frozen even when a parent module
  is unfrozen.
- Activations are not wrapped in `mx::compile` as in Python.

## Tests

```sh
build/bin/mlx_util_test unit
~/.local/mlx/bin/python src/mlx_util/test/python_compat.py build/bin/mlx_util_test
```

`python_compat.py` mirrors `test/zoo.cpp`. It loads Python weights in C++ and
C++ weights in Python and compares the outputs. It also runs every optimizer
for the same steps in both languages and resumes training in each language
from the other's saved optimizer state.
