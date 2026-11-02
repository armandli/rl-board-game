// throughput of a LibTorch MuZero model configuration on this machine:
// batched recurrent inference through ModelInference (with host transfers),
// the bare network forward pass, and learner training steps. Used to size the
// model for the GPU before training.
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <torch/torch.h>

#ifdef RLBG_TORCH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <othello_bitboard.h>

#include <encoder/othello_encoder.h>
#include <experience/fifo_replay_buffer.h>
#include <libtorch/learner.h>
#include <libtorch/model_inference.h>
#include <selfplay/recorder.h>
#include <training/batch.h>

namespace s = std;
namespace r = rlbg;
namespace lt = rlbg::libtorch;
namespace t = torch;

using Game = r::OthelloBitboard;
using Clock = s::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0){
  return s::chrono::duration<double>(Clock::now() - t0).count();
}

// random games for training batches
void fill_replay(const r::OthelloEncoder& enc, r::FifoReplayBuffer& replay, int games, s::mt19937_64& rng){
  r::TrajectoryRecorder<Game> rec(&enc);
  r::ActionList<Game::NUM_ACTIONS> legal;
  for (int i = 0; i < games; ++i){
    Game g;
    auto tr = rec.begin(0);
    while (not g.is_terminal()){
      g.legal_actions(legal);
      r::SearchResult res;
      res.action = legal[s::uniform_int_distribution<int>(0, legal.size - 1)(rng)];
      res.policy.assign(Game::NUM_ACTIONS, 0.f);
      res.policy[res.action] = 1.f;
      res.simulations = 1;
      rec.move(*tr, g, res, true);
    }
    rec.finish(*tr, g);
    replay.add(tr);
  }
}

} // namespace

int main(int argc, char** argv){
  r::OthelloEncoder enc;
  lt::TransformerMuZeroConfig mc;
  mc.num_tokens = enc.num_tokens();
  mc.num_features = enc.num_features();
  mc.num_actions = enc.num_actions();
  mc.num_token_classes = enc.num_token_classes();
  s::vector<int> batches = {256, 1024, 2048, 4096};
  int train_batch = 512;
  int unroll = 5;
  int iterations = 20;
  s::string dtype = "float16";
  s::string optimizer = "muon";
  bool bf16 = true;

  CLI::App app{"LibTorch MuZero throughput benchmark"};
  app.add_option("--dims", mc.dims)->capture_default_str();
  app.add_option("--heads", mc.num_heads)->capture_default_str();
  app.add_option("--repr-layers", mc.repr_layers)->capture_default_str();
  app.add_option("--dyn-layers", mc.dyn_layers)->capture_default_str();
  app.add_option("--pred-layers", mc.pred_layers)->capture_default_str();
  app.add_option("--proj-dims", mc.proj_dims)->capture_default_str();
  app.add_option("--batches", batches, "Inference batch sizes")->delimiter(',')->capture_default_str();
  app.add_option("--train-batch", train_batch, "Training batch size, 0 skips training")->capture_default_str();
  app.add_option("--unroll-steps", unroll)->capture_default_str();
  app.add_option("--iterations", iterations)->capture_default_str();
  app.add_option("--inference-dtype", dtype)->capture_default_str();
  app.add_option("--optimizer", optimizer)->capture_default_str();
  app.add_flag("--bf16,!--no-bf16", bf16, "bfloat16 autocast for training");
  CLI11_PARSE(app, argc, argv);

  t::Device device = lt::parse_device("auto");
  at::globalContext().setAllowTF32CuBLAS(true);
  s::mt19937_64 rng(1);
  t::ScalarType st = lt::parse_dtype(dtype);
  lt::ModelInference inf(mc, device, st);
  {
    lt::TransformerMuZero m(mc);
    s::cout << s::format("model {} with {} parameters on {}", mc.describe(), m->num_parameters(), device.str()) << s::endl;
  }

  for (int b : batches){
    r::InferenceRequests req;
    req.observation_size = inf.observation_size();
    req.latent_size = inf.latent_size();
    s::vector<r::latent_t> lat(inf.latent_size(), r::latent_t(0.5f));
    for (int i = 0; i < b; ++i) req.add_recurrent(lat.data(), i % 65);
    r::InferenceResults res;
    for (int i = 0; i < 3; ++i) inf.infer(req, res);
    auto t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) inf.infer(req, res);
    double full = seconds_since(t0);

    // the network alone, inputs already on the device
    lt::TransformerMuZero model(mc);
    model->to(device, st);
    model->eval();
    t::InferenceMode guard;
    t::Tensor x = t::randn({b, mc.num_tokens, mc.dims}, t::TensorOptions().device(device).dtype(st));
    t::Tensor a = t::randint(0, 65, {b}, t::TensorOptions().device(device).dtype(t::kLong));
    auto net = [&]{
      lt::DynamicsOutput d = model->dynamics(x, a);
      lt::Prediction p = model->predict(d.latent);
      return p.value_logits.sum().item<float>();
    };
    for (int i = 0; i < 3; ++i) net();
    t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) net();
    double bare = seconds_since(t0);
    s::cout << s::format("recurrent batch {:>5}: {:>8.0f} pos/s through ModelInference, {:>8.0f} pos/s network only",
                         b, b * iterations / full, b * iterations / bare) << s::endl;
  }

  if (train_batch > 0){
    r::FifoReplayBuffer replay(1000000);
    fill_replay(enc, replay, 500, rng);
    lt::LearnerConfig lc;
    lc.optimizer = optimizer;
    lc.bf16 = bf16;
    lt::Learner learner(mc, lc, device);
    r::BatchBuilder<Game> builder(&enc, train_batch, unroll, true);
    r::TrainingBatch batch;
    auto t0 = Clock::now();
    double build = 0.;
    for (int i = 0; i < iterations + 3; ++i){
      if (i == 3){
        t0 = Clock::now();
        build = 0.;
      }
      auto b0 = Clock::now();
      builder.build(replay, rng, batch, i);
      build += seconds_since(b0);
      learner.train_step(batch);
    }
    double total = seconds_since(t0);
#ifdef RLBG_TORCH_CUDA
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    s::cout << s::format("peak GPU memory {:.2f} GB", stats.allocated_bytes[0].peak / 1e9) << s::endl;
#endif
    s::cout << s::format("training batch {} unroll {}: {:.1f} steps/s ({:.0f} samples/s), batch building {:.0f}% of the time",
                         train_batch, unroll, iterations / total, iterations * train_batch / total, 100. * build / total)
            << s::endl;
  }
  return 0;
}
