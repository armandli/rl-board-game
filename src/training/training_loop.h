#ifndef RLBG_TRAINING_TRAINING_LOOP_H
#define RLBG_TRAINING_TRAINING_LOOP_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <mlx/mlx.h>

#include <mlx_util/nn.h>

#include <muzero_game.h>

#include <encoder/encoder.h>
#include <evaluation/snapshot_evaluator.h>
#include <experience/binary_experience_store.h>
#include <experience/fifo_replay_buffer.h>
#include <model/model_inference.h>
#include <model/muzero_model.h>
#include <search/mcgs.h>
#include <selfplay/actor.h>
#include <selfplay/inference_server.h>
#include <training/batch.h>
#include <training/muzero_learner.h>
#include <training/sample_ratio.h>

namespace rlbg {

namespace s = std;
namespace sf = std::filesystem;
namespace mx = mlx::core;

struct TrainingLoopConfig {
  s::string data_dir = "data";
  bool      resume = true;
  int64_t   max_steps = 300000;
  uint64_t  seed = 0;                 // 0 picks a random seed

  // learner
  int       batch_size = 256;
  int       unroll_steps = 5;
  bool      augment = true;           // random symmetry per sample
  size_t    replay_capacity = 1000000; // positions
  size_t    min_replay_positions = 50000;
  // training samples drawn per self-play position: the learner waits when
  // it gets ahead and self-play waits when it gets more than
  // selfplay_lead positions ahead, which keeps the ratio of training to
  // self-play fixed whatever the machine's balance of CPU and GPU
  double    sample_reuse = 2.;
  size_t    selfplay_lead = 20000;
  int       publish_interval = 100;   // steps between weight pushes to self-play
  int       snapshot_interval = 2500; // steps between evaluated snapshots
  int       checkpoint_interval = 1000;

  // self-play; 0 picks from the hardware
  int       num_actors = 0;
  int       reanalyze_actors = -1;    // -1: one in eight actors
  s::string inference_dtype = "float16";

  bool      evaluate = true;
  int       log_interval_s = 30;

  SelfPlayConfig        selfplay;
  // EfficientZero off-policy value correction; a game ages out of the full
  // horizon over about horizon * horizon_decay_steps learner steps, roughly
  // the time the replay window covers at the default reuse
  ValueTargetConfig     value{.horizon = 30, .min_horizon = 3, .horizon_decay_steps = 250, .max_value_age = 1000};
  MCGSConfig            search;
  InferenceServerConfig server;
  MuZeroLearnerConfig   learner;
  EvaluatorConfig       evaluator;
};

// MuZero training with every component on its own threads:
//   actors (CPU): self-play games, many per thread, searches batched
//   reanalyze actors (CPU): refresh policy targets of replayed games
//   inference server (GPU): one network call for all actors' requests
//   learner (GPU): samples the replay buffer and trains, pushing weights to
//     the inference server and saving snapshots and checkpoints
//   evaluator (GPU + CPU): rates each snapshot against older snapshots and
//     fixed baselines
// Everything persists under data_dir: experience/ (every game),
// checkpoint/ (model, optimizer, counters), snapshots/, eval/ and
// train_log.csv. With resume set a run continues from the checkpoint and the
// most recent experience.
template <MuZeroGame Game>
struct TrainingLoop {
  using ModelFactory = s::function<s::shared_ptr<MuZeroModel>()>;

  TrainingLoop(const TrainingLoopConfig& config, const Encoder<Game>* encoder, ModelFactory make_model)
  : mConfig(config), mEncoder(encoder), mMakeModel(s::move(make_model))
  {}

  void run(const s::atomic<bool>& interrupt){
    const TrainingLoopConfig& c = mConfig;
    sf::path data(c.data_dir);
    sf::path ckpt = data / "checkpoint";
    sf::path snapshots = data / "snapshots";
    sf::create_directories(ckpt);
    sf::create_directories(snapshots);
    uint64_t seed = c.seed ? c.seed : s::random_device()();
    mx::random::seed(seed);
    s::mt19937_64 rng(seed);

    // learner and its state
    s::shared_ptr<MuZeroModel> model = mMakeModel();
    MuZeroLearner learner(model, c.learner);
    uint64_t previous_positions = 0;
    if (c.resume and learner.load_checkpoint(ckpt.string())){
      s::ifstream(ckpt / "loop.txt") >> previous_positions;
      s::cout << "[train] resumed from step " << learner.step() << s::endl;
    }
    s::cout << "[train] model " << model->type_name() << " with " << model->num_parameters()
            << " parameters" << s::endl;

    FifoReplayBuffer replay(c.replay_capacity);
    BinaryExperienceStore store((data / "experience").string());
    if (c.resume){
      auto trajectories = store.load_recent(c.replay_capacity);
      for (auto& t : trajectories) replay.add(t);
      if (not trajectories.empty())
        s::cout << "[train] loaded " << replay.num_games() << " games, " << replay.num_positions()
                << " positions of experience" << s::endl;
    }
    uint64_t loaded_positions = replay.total_positions();
    // positions of finished games, persisted across runs
    auto generated = [&]{ return previous_positions + replay.total_positions() - loaded_positions; };

    // inference server for the self-play threads
    mx::Dtype dtype = parse_dtype(c.inference_dtype);
    s::shared_ptr<MuZeroModel> infer_model = mMakeModel();
    infer_model->update(learner.weights());
    auto backend = s::make_shared<ModelInference>(infer_model, mEncoder->observation_size(), dtype);
    InferenceServer server(backend, c.server);
    server.set_weights(learner.weights(), (uint32_t)learner.step());

    int latent_size = model->latent_size();
    SearchFactory<Game> make_search = [this, latent_size, sc = c.search](uint64_t seed){
      return s::unique_ptr<Search<Game>>(s::make_unique<MCGS<Game>>(mEncoder, latent_size, sc, seed));
    };

    int hw = (int)s::max(1u, s::thread::hardware_concurrency());
    int actors = c.num_actors > 0 ? c.num_actors : s::max(1, hw - 2 - (c.evaluate ? 1 : 0));
    int reanalyzers = c.reanalyze_actors >= 0 ? s::min(c.reanalyze_actors, actors - 1) : actors / 8;
    s::cout << "[train] " << actors - reanalyzers << " self-play and " << reanalyzers
            << " reanalyze threads x " << c.selfplay.games_per_actor << " games, "
            << c.selfplay.num_simulations << "/" << c.selfplay.fast_simulations << " simulations, inference "
            << c.inference_dtype << ", data in " << data.string() << s::endl;

    s::atomic<bool> stop{false};
    SelfPlayStats stats;
    // the training ratio counts moves as they are played: games run in
    // lockstep batches, so finished positions arrive in bursts
    auto played = [&]{ return (double)(previous_positions + stats.moves.load()); };
    SampleRatio ratio(c.batch_size, c.sample_reuse, c.min_replay_positions, c.selfplay_lead);
    s::vector<s::thread> threads;
    s::thread server_thread([&]{ server.run(); });
    for (int i = 0; i < actors; ++i){
      server.add_client();
      uint64_t aseed = rng();
      if (i < actors - reanalyzers){
        threads.emplace_back([&, aseed]{
          SelfPlayActor<Game> actor(c.selfplay, mEncoder, &server, make_search, &replay, &store, &stats,
                                    [&]{ return server.weights_version(); }, aseed);
          // a paused actor leaves the server's client count so batches do
          // not wait for it
          bool paused = false;
          actor.set_pause([&]{
            bool p = ratio.selfplay_should_pause(learner.step(), played(), replay.num_positions());
            if (p != paused){
              if (p) server.remove_client();
              else   server.add_client();
              paused = p;
            }
            return p;
          });
          actor.run(stop);
          if (not paused) server.remove_client();
        });
      } else {
        threads.emplace_back([&, aseed]{
          ReanalyzeActor<Game> actor(c.selfplay, &server, make_search, &replay, &stats,
                                     [&]{ return server.weights_version(); }, aseed);
          actor.run(stop);
          server.remove_client();
        });
      }
    }

    // evaluator
    s::unique_ptr<SnapshotEvaluator<Game>> evaluator;
    s::thread eval_thread;
    if (c.evaluate){
      evaluator = s::make_unique<SnapshotEvaluator<Game>>(
        c.evaluator, (data / "eval").string(), mMakeModel, mEncoder->observation_size(), dtype, make_search, rng()
      );
      s::vector<sf::path> existing;
      for (const auto& e : sf::directory_iterator(snapshots))
        if (e.path().extension() == ".safetensors") existing.push_back(e.path());
      s::sort(existing.begin(), existing.end());
      for (const sf::path& p : existing) evaluator->add_existing(p.stem().string(), p.string());
      if (existing.empty()){
        sf::path p = snapshots / snapshot_name(learner.step());
        model->save_weights(p.string() + ".safetensors");
        evaluator->submit(p.filename().string(), p.string() + ".safetensors");
      }
      eval_thread = s::thread([&]{ evaluator->run(stop); });
    }

    // learner
    s::mutex log_mutex;
    LearnerStatsAverage averages;
    s::atomic<int64_t> waiting_ms{0};
    auto checkpoint = [&]{
      learner.save_checkpoint(ckpt.string());
      s::ofstream(ckpt / "loop.txt") << generated() << "\n";
      store.flush();
    };
    s::thread learner_thread([&]{
      s::mt19937_64 lrng(rng());
      BatchBuilder<Game> builder(mEncoder, c.batch_size, c.unroll_steps, c.augment, c.value);
      TrainingBatch batch;
      while (not stop.load() and learner.step() < c.max_steps){
        if (not ratio.learner_may_step(learner.step(), played(), replay.num_positions())){
          s::this_thread::sleep_for(s::chrono::milliseconds(20));
          waiting_ms += 20;
          continue;
        }
        builder.build(replay, lrng, batch, learner.step());
        LearnerStats st = learner.train_step(batch);
        {
          s::lock_guard<s::mutex> lock(log_mutex);
          averages.add(st);
        }
        int64_t step = learner.step();
        if (step % c.publish_interval == 0) server.set_weights(learner.weights(), (uint32_t)step);
        if (step % c.checkpoint_interval == 0) checkpoint();
        if (step % c.snapshot_interval == 0){
          sf::path p = snapshots / snapshot_name(step);
          model->save_weights(p.string() + ".safetensors");
          if (evaluator) evaluator->submit(p.filename().string(), p.string() + ".safetensors");
        }
      }
      stop = true;
    });

    // progress log
    sf::path log_path = data / "train_log.csv";
    bool header = not sf::exists(log_path);
    s::ofstream log(log_path, s::ios::app);
    if (header)
      log << "seconds,step,games,positions,replay_positions,reanalyzed,loss,policy,value,reward,"
             "consistency,reconstruction,ownership,grad_norm,lr,inference_positions_per_s,"
             "mean_batch,learner_wait_fraction\n";
    auto t0 = s::chrono::steady_clock::now();
    auto last = t0;
    uint64_t last_games = 0, last_positions = 0, last_nn = 0, last_batches = 0;
    while (not stop.load()){
      for (int i = 0; i < 10 * c.log_interval_s and not stop.load() and not interrupt.load(); ++i)
        s::this_thread::sleep_for(s::chrono::milliseconds(100));
      if (interrupt.load()) break;
      auto now = s::chrono::steady_clock::now();
      double dt = s::chrono::duration<double>(now - last).count();
      double elapsed = s::chrono::duration<double>(now - t0).count();
      last = now;
      LearnerStats avg;
      {
        s::lock_guard<s::mutex> lock(log_mutex);
        avg = averages.take();
      }
      uint64_t games = stats.games, positions = stats.positions;
      uint64_t nn_pos = server.num_positions(), batches = server.num_batches();
      double nn_rate = (nn_pos - last_nn) / dt;
      double mean_batch = batches > last_batches ? (double)(nn_pos - last_nn) / (batches - last_batches) : 0.;
      double wait = s::min(1., waiting_ms.exchange(0) / (1000. * dt));
      s::cout << s::format(
        "[train] {:>6.0f}s step {} | games {} ({:.1f}/s) positions {} ({:.0f}/s) replay {} reanalyzed {} | "
        "loss {:.3f} (p {:.3f} v {:.3f} r {:.3f} c {:.3f} g {:.3f} o {:.3f}) lr {:.2e} | "
        "nn {:.0f} pos/s batch {:.0f} | learner waits {:.0f}%",
        elapsed, learner.step(), games, (games - last_games) / dt, positions, (positions - last_positions) / dt,
        replay.num_positions(), stats.reanalyzed.load(),
        avg.loss, avg.policy, avg.value, avg.reward, avg.consistency, avg.reconstruction, avg.ownership,
        avg.learning_rate, nn_rate, mean_batch, 100. * wait) << s::endl;
      log << s::format("{:.0f},{},{},{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.4f},{:.3e},{:.0f},{:.1f},{:.3f}\n",
        elapsed, learner.step(), games, positions, replay.num_positions(), stats.reanalyzed.load(),
        avg.loss, avg.policy, avg.value, avg.reward, avg.consistency, avg.reconstruction, avg.ownership,
        avg.grad_norm, avg.learning_rate, nn_rate, mean_batch, wait);
      log.flush();
      last_games = games;
      last_positions = positions;
      last_nn = nn_pos;
      last_batches = batches;
    }

    s::cout << "[train] stopping" << s::endl;
    stop = true;
    learner_thread.join();
    server.stop();
    for (s::thread& t : threads) t.join();
    server_thread.join();
    if (eval_thread.joinable()) eval_thread.join();
    checkpoint();
    s::cout << "[train] checkpoint saved at step " << learner.step() << s::endl;
  }

protected:
  static s::string snapshot_name(int64_t step){
    return s::format("step_{:08d}", step);
  }
  TrainingLoopConfig   mConfig;
  const Encoder<Game>* mEncoder;
  ModelFactory         mMakeModel;
};

}  // namespace rlbg

#endif//RLBG_TRAINING_TRAINING_LOOP_H
