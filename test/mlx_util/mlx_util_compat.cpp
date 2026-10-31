// C++ side of python_compat.py, which checks that mlx_util and Python MLX
// read each other's files and compute the same results:
//   python test/mlx_util/python_compat.py build/bin/mlx_util_compat
#include <exception>
#include <iostream>
#include <string>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include "zoo.h"

namespace s = std;
namespace mx = mlx::core;
namespace nn = rlbg::nn;
namespace rt = rlbg::test;

using nn::Params;

namespace {

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
  if (argc < 3) {
    s::cerr << "usage: mlx_util_compat (forward | export | train | resume) DIR" << s::endl;
    return 2;
  }
  s::string mode = argv[1];
  s::string dir = argv[2];
  try {
    if (mode == "forward") return run_forward(dir);
    if (mode == "export") return run_export(dir);
    if (mode == "train") return run_train(dir);
    if (mode == "resume") return run_resume(dir);
  } catch (const s::exception& e) {
    s::cerr << "error: " << e.what() << s::endl;
    return 2;
  }
  s::cerr << "unknown mode " << mode << s::endl;
  return 2;
}
