#include <cmath>
#include <cstdlib>

#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include "zoo.h"

namespace s = std;
namespace sf = std::filesystem;
namespace mx = mlx::core;
namespace nn = rlbg::nn;
namespace optim = rlbg::optim;
namespace rt = rlbg::test;

using nn::Params;

namespace {

int failures = 0;

void check(bool ok, const s::string& what) {
  if (not ok) {
    ++failures;
    s::cout << "FAIL: " << what << s::endl;
  }
}

template <typename F>
void check_throws(F f, const s::string& what) {
  try {
    f();
  } catch (const s::exception&) {
    return;
  }
  check(false, what + " should throw");
}

bool same(const mx::array& a, const mx::array& b) {
  return a.shape() == b.shape() and mx::array_equal(a, b).item<bool>();
}

bool close(const mx::array& a, const mx::array& b, double tol = 1e-5) {
  return a.shape() == b.shape() and mx::allclose(a, b, tol, tol).item<bool>();
}

s::vector<s::string> keys_of(const Params& p) {
  s::vector<s::string> out;
  for (const auto& [k, v] : p) out.push_back(k);
  return out;
}

s::string tmp_dir() {
  sf::path p = sf::temp_directory_path() / "mlx_util_test";
  sf::create_directories(p);
  return p.string();
}

void test_module_tree() {
  auto mlp = s::make_shared<nn::Sequential>(
    s::make_shared<nn::Linear>(8, 16),
    nn::relu,
    s::make_shared<nn::Linear>(16, 4, false)
  );
  check(
    keys_of(mlp->parameters()) ==
      s::vector<s::string>{"layers.0.bias", "layers.0.weight", "layers.2.weight"},
    "Sequential parameter keys skip functions but keep their index"
  );
  s::ostringstream os;
  os << *mlp;
  check(
    os.str() ==
      "Sequential(\n"
      "  (layers.0): Linear(input_dims=8, output_dims=16, bias=True)\n"
      "  (layers.1): Lambda()\n"
      "  (layers.2): Linear(input_dims=16, output_dims=4, bias=False)\n"
      ")",
    "repr matches Python layout: got\n" + os.str()
  );
  check(mlp->named_modules().size() == 5, "named_modules counts root, list and layers");
  check(mlp->leaf_modules().size() == 3, "leaf_modules");
  check((*mlp)(mx::ones({2, 8})).shape() == mx::Shape{2, 4}, "Sequential forward shape");

  mlp->freeze();
  check(mlp->trainable_parameters().empty(), "freeze() freezes everything");
  mlp->unfreeze();
  check(mlp->trainable_parameters().size() == 3, "unfreeze() restores everything");
  mlp->freeze(s::vector<s::string>{"bias"});
  check(
    keys_of(mlp->trainable_parameters()) ==
      s::vector<s::string>{"layers.0.weight", "layers.2.weight"},
    "freeze(keys) recurses into children"
  );
  mlp->unfreeze();
  mlp->freeze(s::vector<s::string>{"layers"}, false);
  check(mlp->trainable_parameters().empty(), "freezing a child name freezes the subtree");
  mlp->unfreeze();
  check_throws([&] { mlp->freeze(s::vector<s::string>{"nope"}, true, true); }, "strict freeze of unknown key");

  check_throws([&] { mlp->update({{"layers.9.weight", mx::zeros({1})}}); }, "strict update of unknown key");
  mlp->update({{"layers.9.weight", mx::zeros({1})}}, false);
  mlp->update({{"layers.0.bias", mx::zeros({16})}});
  check(same(mlp->parameters().at("layers.0.bias"), mx::zeros({16})), "update assigns by flat key");

  Params good = mlp->parameters();
  Params extra = good;
  extra.insert_or_assign("layers.5.weight", mx::zeros({1}));
  Params missing = good;
  missing.erase("layers.2.weight");
  Params wrong = good;
  wrong.insert_or_assign("layers.0.bias", mx::zeros({3}));
  check_throws([&] { mlp->load_weights(extra); }, "load_weights with extra key");
  check_throws([&] { mlp->load_weights(missing); }, "load_weights with missing key");
  check_throws([&] { mlp->load_weights(wrong); }, "load_weights with wrong shape");
  mlp->load_weights(missing, false);

  auto bn = s::make_shared<nn::Sequential>(s::make_shared<nn::Linear>(4, 4), s::make_shared<nn::BatchNorm>(4));
  check(bn->parameters().size() == 6, "BatchNorm running stats are parameters");
  check(bn->trainable_parameters().size() == 4, "BatchNorm running stats are frozen");
  bn->unfreeze();
  check(bn->trainable_parameters().size() == 4, "BatchNorm running stats stay frozen after unfreeze");

  auto dict = s::make_shared<nn::ModuleDict>();
  dict->insert("policy", s::make_shared<nn::Linear>(4, 2));
  dict->insert("value", s::make_shared<nn::Linear>(4, 1));
  check(
    keys_of(dict->parameters()) ==
      s::vector<s::string>{"policy.bias", "policy.weight", "value.bias", "value.weight"},
    "ModuleDict keys"
  );
  check_throws([&] { dict->at<nn::Conv1d>("policy"); }, "ModuleDict::at with wrong type");
}

void test_io() {
  s::string dir = tmp_dir();
  Params arrays{
    {"a.weight", mx::random::normal({3, 4})},
    {"b", mx::arange(10)},
    {"c.half", mx::astype(mx::random::normal({5}), mx::float16)},
    {"d.bf16", mx::astype(mx::random::normal({2, 2}), mx::bfloat16)},
    {"e.scalar", mx::array(7, mx::uint64)},
  };
  for (const char* name : {"/a.npz", "/a.safetensors"}) {
    nn::save_arrays(dir + name, arrays);
    Params back = nn::load_arrays(dir + name);
    check(keys_of(back) == keys_of(arrays), s::string("keys roundtrip ") + name);
    for (const auto& [k, v] : arrays)
      check(back.contains(k) and same(back.at(k), v) and back.at(k).dtype() == v.dtype(),
            s::string("value roundtrip ") + name + " " + k);
  }
  if (nn::has_npz_compression()) {
    nn::save_npz(dir + "/c.npz", arrays, true);
    Params back = nn::load_npz(dir + "/c.npz");
    for (const auto& [k, v] : arrays)
      check(same(back.at(k), v), "compressed npz roundtrip " + k);
  }
  check_throws([&] { nn::save_arrays(dir + "/a.bin", arrays); }, "unsupported extension");
}

void test_value_and_grad() {
  mx::random::seed(3);
  auto model = s::make_shared<nn::Linear>(3, 1);
  mx::array x = mx::random::normal({64, 3});
  mx::array y = mx::matmul(x, mx::array({1.5f, -2.0f, 0.5f}, {3, 1})) + mx::array(0.3f);
  optim::SGD opt(0.1);
  auto step = nn::value_and_grad(*model, [&](const mx::array& xb, const mx::array& yb) {
    return nn::mse_loss((*model)(xb), yb);
  });
  float first = 0;
  float last = 0;
  for (int i = 0; i < 200; ++i) {
    Params before = model->parameters();
    auto [loss, grads] = step(x, y);
    check(keys_of(grads) == keys_of(model->trainable_parameters()), "grad keys");
    if (i == 0) {
      for (const auto& [k, v] : before)
        check(same(model->parameters().at(k), v), "value_and_grad restores parameters");
      first = loss.item<float>();
    }
    opt.update(*model, grads);
    nn::eval(model->parameters());
    last = loss.item<float>();
  }
  check(last < 1e-4f and last < first, "linear regression converges: " + s::to_string(last));

  model->freeze(s::vector<s::string>{"bias"});
  auto [loss, grads] = nn::value_and_grad(*model, [&] { return nn::mse_loss((*model)(x), y); })();
  check(keys_of(grads) == s::vector<s::string>{"weight"}, "frozen parameters get no gradient");
}

void test_checkpoint() {
  mx::random::seed(4);
  auto plain = s::make_shared<nn::TransformerEncoder>(2, 8, 2, 16);
  auto ckpt = s::make_shared<nn::TransformerEncoder>(2, 8, 2, 16, 0.0, nullptr, true, true);
  ckpt->load_weights(plain->parameters());
  mx::array x = mx::random::normal({2, 5, 8});
  nn::Mask mask = nn::MultiHeadAttention::create_additive_causal_mask(5);
  auto [l1, g1] = nn::value_and_grad(*plain, [&] { return mx::sum(plain->forward(x, mask)); })();
  auto [l2, g2] = nn::value_and_grad(*ckpt, [&] { return mx::sum(ckpt->forward(x, mask)); })();
  check(close(l1, l2, 1e-4), "checkpointed loss matches");
  for (const auto& [k, v] : g1)
    check(g2.contains(k) and close(v, g2.at(k), 1e-4), "checkpointed gradient matches " + k);
}

void test_optimizer_state() {
  s::string dir = tmp_dir();
  mx::random::seed(5);
  auto model = rt::build_train_model();
  Params inputs{{"train.x", mx::random::normal({16, 8})}, {"train.y", mx::random::normal({16, 4})}};
  for (const rt::OptEntry& entry : rt::optimizers()) {
    auto m1 = rt::build_train_model();
    m1->load_weights(model->parameters());
    auto opt = entry.build();
    for (int i = 0; i < 2; ++i) rt::train_step(*m1, *opt, entry, inputs);
    opt->save_state(dir + "/opt.safetensors");

    auto m2 = rt::build_train_model();
    m2->load_weights(m1->parameters());
    auto opt2 = entry.build();
    opt2->load_state(dir + "/opt.safetensors");
    check(keys_of(opt2->state()) == keys_of(opt->state()), entry.name + " state keys roundtrip");
    rt::train_step(*m1, *opt, entry, inputs);
    rt::train_step(*m2, *opt2, entry, inputs);
    for (const auto& [k, v] : m1->parameters())
      check(close(v, m2->parameters().at(k), 1e-6), entry.name + " resumes identically " + k);
  }
  auto multi = optim::MultiOptimizer(
    {s::make_shared<optim::Adam>(1e-2), s::make_shared<optim::SGD>(0.1)},
    {[](const s::string& k, const mx::array&) { return k.starts_with("layers.0."); }}
  );
  check(keys_of(multi.state()) == s::vector<s::string>{"states.0.learning_rate", "states.0.step",
                                                        "states.1.learning_rate", "states.1.step"},
        "MultiOptimizer state keys");
}

void test_layers_misc() {
  mx::random::seed(6);
  auto mlp = s::make_shared<nn::Sequential>(
    s::make_shared<nn::Linear>(4, 8), s::make_shared<nn::LayerNorm>(8),
    s::make_shared<nn::GELU>("tanh"), s::make_shared<nn::Linear>(8, 2)
  );
  mlp->set_dtype(mx::float16);
  for (const auto& [k, v] : mlp->parameters()) check(v.dtype() == mx::float16, "set_dtype " + k);
  check((*mlp)(mx::ones({3, 4}, mx::float16)).dtype() == mx::float16, "float16 forward stays float16");

  nn::Dropout drop(0.5);
  mx::array ones = mx::ones({1000});
  check(not same(drop(ones), ones), "dropout drops in training");
  drop.eval();
  check(same(drop(ones), ones), "dropout is identity in eval");

  mx::array w = mx::zeros({64, 32});
  mx::array g = nn::init::glorot_uniform()(w);
  double limit = std::sqrt(6.0 / 96);
  check(g.shape() == w.shape() and mx::max(mx::abs(g)).item<float>() <= limit + 1e-6, "glorot_uniform bound");
  mx::array q = nn::init::orthogonal()(mx::zeros({16, 16}));
  // GPU matmuls run in TF32 on CUDA, so QᵀQ is only identity to ~1e-3
  check(close(mx::matmul(mx::transpose(q), q), mx::eye(16), 2e-3), "orthogonal init");
  mx::array sp = nn::init::sparse(0.25)(mx::zeros({10, 8}));
  check(mx::sum(mx::equal(sp, mx::array(0.0f))).item<int>() == 20, "sparse init zeros per row");
  check(same(nn::init::constant(2.0)(w), mx::full({64, 32}, 2.0f)), "constant init");
  check(nn::init::he_normal(mx::float32, "fan_out")(w).shape() == w.shape(), "he_normal shape");

  Params grads{{"a", mx::full({4}, 3.0f)}, {"b", mx::full({4}, 4.0f)}};
  auto [clipped, norm] = optim::clip_grad_norm(grads, 1.0);
  check(std::abs(norm.item<float>() - 10.0f) < 1e-4, "clip_grad_norm reports norm");
  check(close(clipped.at("a"), mx::full({4}, 0.3f), 1e-4), "clip_grad_norm scales");

  check_throws([] { nn::Conv2d(3, 4, nn::Int2{1, 2, 3}); }, "IntN rejects wrong arity");
  check_throws([] { nn::GroupNorm(3, 4); }, "GroupNorm divisibility");
  check(nn::MaxPool2d(2)(mx::ones({1, 5, 5, 1})).shape() == mx::Shape{1, 2, 2, 1}, "strided pooling");
}

int run_unit() {
  test_module_tree();
  test_io();
  test_value_and_grad();
  test_checkpoint();
  test_optimizer_state();
  test_layers_misc();
  s::cout << (failures ? "unit tests FAILED: " + s::to_string(failures) : s::string("unit tests passed"))
          << s::endl;
  return failures ? 1 : 0;
}

void save_weights(nn::Module& m, const s::string& file_base, const s::string& format) {
  if (format == ".npz.compressed") nn::save_npz(file_base + ".npz", m.parameters(), true);
  else m.save_weights(file_base + format);
}

s::string weights_file(const s::string& base, const s::string& format) {
  return base + (format == ".npz.compressed" ? s::string(".npz") : format);
}

// loads Python's zoo weights and writes this side's outputs
int run_forward(const s::string& dir) {
  Params inputs = nn::load_arrays(dir + "/inputs.npz");
  Params out;
  for (const rt::ZooEntry& e : rt::zoo()) {
    auto m = e.build();
    m->load_weights(weights_file(dir + "/py_" + e.name, e.py_format));
    for (auto& [k, v] : e.run(*m, inputs)) out.insert_or_assign(k, v);
  }
  for (auto& [k, v] : rt::functions_outputs(inputs)) out.insert_or_assign(k, v);
  nn::save_npz(dir + "/cpp_forward.npz", out);
  return 0;
}

// C++ initialized zoo for Python to load
int run_export(const s::string& dir) {
  Params inputs = nn::load_arrays(dir + "/inputs.npz");
  Params out;
  mx::random::seed(11);
  for (const rt::ZooEntry& e : rt::zoo()) {
    auto m = e.build();
    if (e.warmup) e.warmup(*m, inputs);
    save_weights(*m, dir + "/cpp_" + e.name, e.cpp_format);
    for (auto& [k, v] : e.run(*m, inputs)) out.insert_or_assign(k, v);
  }
  nn::save_arrays(dir + "/cpp_export.safetensors", out);
  return 0;
}

int run_train(const s::string& dir) {
  Params inputs = nn::load_arrays(dir + "/inputs.npz");
  for (const rt::OptEntry& entry : rt::optimizers()) {
    auto m = rt::build_train_model();
    m->load_weights(dir + "/train_w0.safetensors");
    auto opt = entry.build();
    for (int i = 0; i < 3; ++i) rt::train_step(*m, *opt, entry, inputs);
    m->save_weights(dir + "/cpp_" + entry.name + "_w3.npz");
    opt->save_state(dir + "/cpp_" + entry.name + "_s3.safetensors");
    rt::train_step(*m, *opt, entry, inputs);
    m->save_weights(dir + "/cpp_" + entry.name + "_w4.npz");
  }
  return 0;
}

int run_resume(const s::string& dir) {
  Params inputs = nn::load_arrays(dir + "/inputs.npz");
  for (const rt::OptEntry& entry : rt::optimizers()) {
    auto m = rt::build_train_model();
    m->load_weights(dir + "/py_" + entry.name + "_w3.npz");
    auto opt = entry.build();
    opt->load_state(dir + "/py_" + entry.name + "_s3.safetensors");
    rt::train_step(*m, *opt, entry, inputs);
    m->save_weights(dir + "/cpp_" + entry.name + "_resume_w4.safetensors");
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  s::string mode = argc > 1 ? argv[1] : "unit";
  s::string dir = argc > 2 ? argv[2] : "";
  try {
    if (mode == "unit") return run_unit();
    if (mode == "forward") return run_forward(dir);
    if (mode == "export") return run_export(dir);
    if (mode == "train") return run_train(dir);
    if (mode == "resume") return run_resume(dir);
  } catch (const s::exception& e) {
    s::cerr << "error: " << e.what() << s::endl;
    return 2;
  }
  s::cerr << "usage: mlx_util_test [unit | forward DIR | export DIR | train DIR | resume DIR]" << s::endl;
  return 2;
}
