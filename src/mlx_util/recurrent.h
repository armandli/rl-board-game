#ifndef RLBG_MLX_UTIL_RECURRENT_H
#define RLBG_MLX_UTIL_RECURRENT_H

#include <optional>
#include <string>
#include <utility>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// inputs are (N, L, D) or (L, D); outputs hold every step's hidden state

struct RNN : Module {
  // nonlinearity defaults to tanh
  RNN(int input_size, int hidden_size, bool bias = true, ArrayFn nonlinearity = nullptr);

  s::string type_name() const override { return "RNN"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x, s::optional<mx::array> hidden = s::nullopt);
  mx::array operator()(const mx::array& x, s::optional<mx::array> hidden = s::nullopt) {
    return forward(x, s::move(hidden));
  }

  mx::array Wxh;
  mx::array Whh;
  s::optional<mx::array> bias;
  ArrayFn nonlinearity;
  int hidden_size;
};

struct GRU : Module {
  GRU(int input_size, int hidden_size, bool bias = true);

  s::string type_name() const override { return "GRU"; }
  s::string extra_repr() const override;
  mx::array forward(const mx::array& x, s::optional<mx::array> hidden = s::nullopt);
  mx::array operator()(const mx::array& x, s::optional<mx::array> hidden = s::nullopt) {
    return forward(x, s::move(hidden));
  }

  mx::array Wx;
  mx::array Wh;
  s::optional<mx::array> b;
  s::optional<mx::array> bhn;
  int hidden_size;
};

// returns {hidden states, cell states}
struct LSTM : Module {
  LSTM(int input_size, int hidden_size, bool bias = true);

  s::string type_name() const override { return "LSTM"; }
  s::string extra_repr() const override;
  s::pair<mx::array, mx::array> forward(
    const mx::array& x,
    s::optional<mx::array> hidden = s::nullopt,
    s::optional<mx::array> cell = s::nullopt
  );
  s::pair<mx::array, mx::array> operator()(
    const mx::array& x,
    s::optional<mx::array> hidden = s::nullopt,
    s::optional<mx::array> cell = s::nullopt)
  {
    return forward(x, s::move(hidden), s::move(cell));
  }

  mx::array Wx;
  mx::array Wh;
  s::optional<mx::array> bias;
  int hidden_size;
};

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_RECURRENT_H
