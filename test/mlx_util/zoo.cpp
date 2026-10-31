#include "zoo.h"

#include <utility>

namespace rlbg::test {

namespace {

using nn::Mask;

mx::array in(const Params& inputs, const s::string& key) { return inputs.at(key); }

struct Recurrent : nn::Module {
  Recurrent()
    : rnn(register_module("rnn", s::make_shared<nn::RNN>(6, 8))),
      gru(register_module("gru", s::make_shared<nn::GRU>(8, 8))),
      lstm(register_module("lstm", s::make_shared<nn::LSTM>(8, 4))) {}

  s::string type_name() const override { return "Recurrent"; }
  mx::array forward(const mx::array& x) {
    mx::array h = gru->forward(rnn->forward(x));
    auto [hidden, cell] = lstm->forward(h);
    return hidden + cell;
  }

  s::shared_ptr<nn::RNN> rnn;
  s::shared_ptr<nn::GRU> gru;
  s::shared_ptr<nn::LSTM> lstm;
};

struct Transformers : nn::Module {
  Transformers()
    : enc(register_module("enc", s::make_shared<nn::TransformerEncoder>(2, 16, 4, 32))),
      tr(register_module("tr", s::make_shared<nn::Transformer>(
        16, 4, 1, 1, 32, 0.0, nullptr, nullptr, nullptr, false
      ))) {}

  s::string type_name() const override { return "Transformers"; }

  s::shared_ptr<nn::TransformerEncoder> enc;
  s::shared_ptr<nn::Transformer> tr;
};

struct Attention : nn::Module {
  Attention()
    : emb(register_module("emb", s::make_shared<nn::Embedding>(10, 16))),
      pe(register_module("pe", s::make_shared<nn::SinusoidalPositionalEncoding>(16))),
      mha(register_module("mha", s::make_shared<nn::MultiHeadAttention>(
        16, 4, s::nullopt, s::nullopt, s::nullopt, s::nullopt, 8, true
      ))),
      rope(register_module("rope", s::make_shared<nn::RoPE>(4))),
      bil(register_module("bil", s::make_shared<nn::Bilinear>(16, 8, 5))) {}

  s::string type_name() const override { return "Attention"; }

  s::shared_ptr<nn::Embedding> emb;
  s::shared_ptr<nn::SinusoidalPositionalEncoding> pe;
  s::shared_ptr<nn::MultiHeadAttention> mha;
  s::shared_ptr<nn::RoPE> rope;
  s::shared_ptr<nn::Bilinear> bil;
};

Params eval_only(const s::string& name, nn::Module& m, const s::function<mx::array()>& f) {
  m.eval();
  return {{name + ".out", f()}};
}

} // namespace

const s::vector<ZooEntry>& zoo() {
  static const s::vector<ZooEntry> entries = {
    {
      "mlp", ".npz", ".safetensors",
      [] {
        return s::make_shared<nn::Sequential>(
          s::make_shared<nn::Linear>(8, 16),
          nn::relu,
          s::make_shared<nn::Linear>(16, 16),
          s::make_shared<nn::GELU>(),
          s::make_shared<nn::Dropout>(0.1),
          s::make_shared<nn::Linear>(16, 4)
        );
      },
      nullptr,
      [](nn::Module& m, const Params& x) {
        auto& seq = dynamic_cast<nn::Sequential&>(m);
        return eval_only("mlp", m, [&] { return seq(in(x, "mlp.x")); });
      },
    },
    {
      "conv2d", ".safetensors", ".npz",
      [] {
        return s::make_shared<nn::Sequential>(
          s::make_shared<nn::Conv2d>(3, 8, 3, 1, 1),
          s::make_shared<nn::BatchNorm>(8),
          s::make_shared<nn::ReLU>(),
          s::make_shared<nn::MaxPool2d>(2),
          s::make_shared<nn::Conv2d>(8, 4, nn::Int2{3, 3}, 2, 1),
          s::make_shared<nn::GroupNorm>(2, 4),
          s::make_shared<nn::AvgPool2d>(2, 1, 1),
          s::make_shared<nn::Upsample>(2.0, "linear"),
          s::make_shared<nn::ConvTranspose2d>(4, 3, 3, 2, 1, 1, 1)
        );
      },
      [](nn::Module& m, const Params& x) {
        m.train();
        mx::eval(dynamic_cast<nn::Sequential&>(m)(in(x, "conv2d.x")));
        nn::eval(m.parameters());
      },
      [](nn::Module& m, const Params& x) {
        auto& seq = dynamic_cast<nn::Sequential&>(m);
        Params out;
        m.eval();
        out.insert_or_assign("conv2d.eval", seq(in(x, "conv2d.x")));
        m.train();
        out.insert_or_assign("conv2d.train", seq(in(x, "conv2d.x")));
        auto bn = s::dynamic_pointer_cast<nn::BatchNorm>(seq[1]);
        out.insert_or_assign("conv2d.running_mean", *bn->running_mean);
        out.insert_or_assign("conv2d.running_var", *bn->running_var);
        return out;
      },
    },
    {
      "conv1d", ".npz.compressed", ".npz",
      [] {
        return s::make_shared<nn::Sequential>(
          s::make_shared<nn::Conv1d>(4, 6, 3, 1, 1, 1, 2),
          s::make_shared<nn::ConvTranspose1d>(6, 5, 3, 2),
          s::make_shared<nn::InstanceNorm>(5, 1e-5f, true),
          s::make_shared<nn::MaxPool1d>(2),
          s::make_shared<nn::LayerNorm>(5),
          s::make_shared<nn::AvgPool1d>(3, 2),
          s::make_shared<nn::RMSNorm>(5),
          s::make_shared<nn::Conv1d>(5, 4, 2, 1, 0, 2)
        );
      },
      nullptr,
      [](nn::Module& m, const Params& x) {
        auto& seq = dynamic_cast<nn::Sequential&>(m);
        return eval_only("conv1d", m, [&] { return seq(in(x, "conv1d.x")); });
      },
    },
    {
      "conv3d", ".safetensors", ".npz.compressed",
      [] {
        return s::make_shared<nn::Sequential>(
          s::make_shared<nn::Conv3d>(2, 3, 3, 1, 1),
          s::make_shared<nn::ConvTranspose3d>(3, 2, 2, 2),
          s::make_shared<nn::MaxPool3d>(2),
          s::make_shared<nn::AvgPool3d>(nn::Int3{2, 2, 2}, nn::Int3{1, 1, 1}),
          s::make_shared<nn::Upsample>(s::vector<double>{2.0, 1.0, 1.5})
        );
      },
      nullptr,
      [](nn::Module& m, const Params& x) {
        auto& seq = dynamic_cast<nn::Sequential&>(m);
        return eval_only("conv3d", m, [&] { return seq(in(x, "conv3d.x")); });
      },
    },
    {
      "recurrent", ".npz", ".safetensors",
      [] { return s::make_shared<Recurrent>(); },
      nullptr,
      [](nn::Module& m, const Params& x) {
        auto& r = dynamic_cast<Recurrent&>(m);
        return eval_only("recurrent", m, [&] { return r.forward(in(x, "recurrent.x")); });
      },
    },
    {
      "transformer", ".safetensors", ".npz",
      [] { return s::make_shared<Transformers>(); },
      nullptr,
      [](nn::Module& m, const Params& x) {
        auto& t = dynamic_cast<Transformers&>(m);
        m.eval();
        mx::array src = in(x, "transformer.src");
        mx::array tgt = in(x, "transformer.tgt");
        Mask src_mask = nn::MultiHeadAttention::create_additive_causal_mask(src.shape(1));
        Mask tgt_mask = nn::MultiHeadAttention::create_additive_causal_mask(tgt.shape(1));
        return Params{
          {"transformer.enc", t.enc->forward(src, src_mask)},
          {"transformer.tr", t.tr->forward(src, tgt, s::nullopt, tgt_mask, s::nullopt)},
        };
      },
    },
    {
      "attention", ".npz", ".safetensors",
      [] { return s::make_shared<Attention>(); },
      nullptr,
      [](nn::Module& m, const Params& x) {
        auto& a = dynamic_cast<Attention&>(m);
        m.eval();
        mx::array tokens = in(x, "attention.tokens");
        mx::array e = a.emb->forward(tokens) + a.pe->forward(in(x, "attention.pos"));
        Mask mask = nn::MultiHeadAttention::create_additive_causal_mask(tokens.shape(1));
        mx::array h = a.mha->forward(e, e, e, mask);
        mx::array r = a.rope->forward(h, 2);
        return Params{
          {"attention.bil", a.bil->forward(e, r)},
          {"attention.tied", a.emb->as_linear(e)},
        };
      },
    },
  };
  return entries;
}

Params functions_outputs(const Params& x) {
  Params out;
  auto put = [&](const s::string& k, const mx::array& v) { out.insert_or_assign("fn." + k, v); };

  s::vector<nn::UnaryModulePtr> acts = {
    s::make_shared<nn::Sigmoid>(), s::make_shared<nn::Mish>(), s::make_shared<nn::ReLU>(),
    s::make_shared<nn::ReLU2>(), s::make_shared<nn::ReLU6>(), s::make_shared<nn::Softmax>(),
    s::make_shared<nn::Softplus>(), s::make_shared<nn::Softsign>(), s::make_shared<nn::SiLU>(),
    s::make_shared<nn::LogSoftmax>(), s::make_shared<nn::LogSigmoid>(), s::make_shared<nn::Tanh>(),
    s::make_shared<nn::Hardswish>(), s::make_shared<nn::SELU>(), s::make_shared<nn::HardTanh>(),
    s::make_shared<nn::Softmin>(), s::make_shared<nn::GLU>(), s::make_shared<nn::LeakyReLU>(0.2),
    s::make_shared<nn::ELU>(0.5), s::make_shared<nn::CELU>(0.7), s::make_shared<nn::Softshrink>(0.3),
    s::make_shared<nn::HardShrink>(0.3), s::make_shared<nn::Step>(0.1), s::make_shared<nn::PReLU>(8),
    s::make_shared<nn::GELU>(), s::make_shared<nn::GELU>("precise"), s::make_shared<nn::GELU>("fast"),
  };
  mx::array ax = x.at("fn.act_x");
  for (size_t i = 0; i < acts.size(); ++i) put("act." + s::to_string(i), acts[i]->forward(ax));

  using nn::Reduction;
  mx::array logits = x.at("fn.logits");
  mx::array targets = x.at("fn.targets");
  mx::array probs = x.at("fn.probs");
  mx::array w = x.at("fn.weights");
  mx::array a = x.at("fn.a");
  mx::array b = x.at("fn.b");
  mx::array b01 = x.at("fn.b01");
  mx::array sgn = x.at("fn.sign");
  mx::array vars = x.at("fn.vars");
  put("ce", nn::cross_entropy(logits, targets));
  put("ce_ls", nn::cross_entropy(logits, targets, s::nullopt, -1, 0.1, Reduction::Mean));
  put("ce_p", nn::cross_entropy(logits, probs, w, -1, 0.0, Reduction::Sum));
  put("bce", nn::binary_cross_entropy(a, b01));
  put("bce_nl", nn::binary_cross_entropy(mx::sigmoid(a), b01, s::nullopt, false, Reduction::None));
  put("l1", nn::l1_loss(a, b));
  put("mse", nn::mse_loss(a, b, Reduction::Sum));
  put("nll", nn::nll_loss(nn::log_softmax(logits), targets, -1, Reduction::Mean));
  put("gnll", nn::gaussian_nll_loss(a, b, vars, true));
  put("kl", nn::kl_div_loss(nn::log_softmax(a), nn::log_softmax(b)));
  put("smooth_l1", nn::smooth_l1_loss(a, b, 0.5));
  put("triplet", nn::triplet_loss(a, b, x.at("fn.c")));
  put("hinge", nn::hinge_loss(a, sgn));
  put("huber", nn::huber_loss(a, b, 0.5));
  put("log_cosh", nn::log_cosh_loss(a, b));
  put("cosine", nn::cosine_similarity_loss(a, b, 1));
  put("margin", nn::margin_ranking_loss(a, b, sgn, 0.1));

  mx::array img = x.at("fn.img");
  put("up_nearest", nn::Upsample(2.0).forward(img));
  put("up_nearest_frac", nn::Upsample(s::vector<double>{1.5, 2.5}).forward(img));
  put("up_linear", nn::Upsample(s::vector<double>{2.0, 3.0}, "linear").forward(img));
  put("up_linear_ac", nn::Upsample(2.0, "linear", true).forward(img));
  put("up_linear_aa", nn::Upsample(s::vector<double>{0.5, 0.75}, "linear", false, true).forward(img));
  put("up_cubic", nn::Upsample(2.0, "cubic").forward(img));
  put("up_cubic_ac", nn::Upsample(2.0, "cubic", true).forward(img));
  put("up_cubic_aa", nn::Upsample(0.5, "cubic", false, true).forward(img));
  put("up_1d", nn::Upsample(1.5, "linear").forward(x.at("fn.seq")));

  mx::array scores = x.at("fn.scores");
  put("alibi", nn::ALiBi().forward(scores));
  put("alibi_offset", nn::ALiBi().forward(scores, 2, x.at("fn.score_mask")));
  put("alibi_slope6", nn::ALiBi::create_alibi_slope(6));
  put("causal", nn::MultiHeadAttention::create_additive_causal_mask(5));
  put("spe_cos", nn::SinusoidalPositionalEncoding(8, 1e-3, 1.0, 1.0, true, true).forward(x.at("fn.pos")));
  return out;
}

nn::ModulePtr build_train_model() {
  return s::make_shared<nn::Sequential>(
    s::make_shared<nn::Linear>(8, 32),
    s::make_shared<nn::ReLU>(),
    s::make_shared<nn::Linear>(32, 32),
    s::make_shared<nn::Tanh>(),
    s::make_shared<nn::Linear>(32, 4)
  );
}

mx::array train_loss(nn::Module& model, const Params& inputs) {
  auto& seq = dynamic_cast<nn::Sequential&>(model);
  return nn::mse_loss(seq(inputs.at("train.x")), inputs.at("train.y"));
}

const s::vector<OptEntry>& optimizers() {
  using namespace optim;
  static const s::vector<OptEntry> entries = {
    {"sgd", [] { return s::make_shared<SGD>(0.05, 0.9, 1e-3, 0.0, true); }},
    {"sgd_plain", [] { return s::make_shared<SGD>(0.05); }},
    {"sgd_damp", [] { return s::make_shared<SGD>(0.05, 0.9, 0.0, 0.2); }},
    {"rmsprop", [] { return s::make_shared<RMSprop>(1e-2); }},
    {"adagrad", [] { return s::make_shared<Adagrad>(0.05); }},
    {"adadelta", [] { return s::make_shared<AdaDelta>(1.0); }},
    {"adam", [] { return s::make_shared<Adam>(1e-2, s::pair{0.9, 0.999}, 1e-8, true); }},
    {"adamw_cos", [] { return s::make_shared<AdamW>(cosine_decay(1e-2, 10), s::pair{0.9, 0.999}, 1e-8, 0.05); }},
    {"adamax", [] { return s::make_shared<Adamax>(1e-2); }},
    {"lion", [] { return s::make_shared<Lion>(1e-3, s::pair{0.9, 0.99}, 0.01); }},
    {"adafactor", [] { return s::make_shared<Adafactor>(); }},
    {"adafactor_b1", [] {
      return s::make_shared<Adafactor>(1e-2, s::pair{1e-30, 1e-3}, 1.0, -0.8, 0.9, 0.01, false, false);
    }},
    {"muon", [] { return s::make_shared<Muon>(1e-2); }},
    {"sgd_joined", [] {
      return s::make_shared<SGD>(
        join_schedules({linear_schedule(0.0, 0.1, 2), exponential_decay(0.1, 0.9)}, {2}),
        0.5
      );
    }},
    {"sgd_step_decay", [] { return s::make_shared<SGD>(step_decay(0.1, 0.5, 2)); }},
    {"multi", [] {
      return s::make_shared<MultiOptimizer>(
        s::vector<s::shared_ptr<Optimizer>>{s::make_shared<Adam>(1e-2), s::make_shared<SGD>(0.05)},
        s::vector<MultiOptimizer::Filter>{
          [](const s::string& k, const mx::array&) { return k.ends_with("bias"); },
        }
      );
    }},
    {"adam_clip", [] { return s::make_shared<Adam>(1e-2); }, 0.5},
  };
  return entries;
}

void train_step(nn::Module& model, optim::Optimizer& opt, const OptEntry& entry, const Params& inputs) {
  auto loss_and_grad = nn::value_and_grad(model, [&] { return train_loss(model, inputs); });
  auto [loss, grads] = loss_and_grad();
  if (entry.clip > 0) grads = optim::clip_grad_norm(grads, entry.clip).first;
  opt.update(model, grads);
  nn::eval(model.parameters());
  nn::eval(opt.state());
}

} // namespace rlbg::test
