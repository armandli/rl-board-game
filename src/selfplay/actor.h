#ifndef RLBG_SELFPLAY_ACTOR_H
#define RLBG_SELFPLAY_ACTOR_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <muzero_game.h>

#include <encoder/encoder.h>
#include <experience/experience_store.h>
#include <experience/replay_buffer.h>
#include <search/inference.h>
#include <search/search.h>
#include <selfplay/recorder.h>

namespace rlbg {

namespace s = std;

template <typename Game>
using SearchFactory = s::function<s::unique_ptr<Search<Game>>(uint64_t seed)>;

struct SelfPlayStats {
  s::atomic<uint64_t> games{0};
  s::atomic<uint64_t> positions{0}; // of finished games
  s::atomic<uint64_t> moves{0};     // played, including unfinished games
  s::atomic<uint64_t> full_searches{0};
  s::atomic<uint64_t> wins{0};      // player 0
  s::atomic<uint64_t> losses{0};
  s::atomic<uint64_t> draws{0};
  s::atomic<uint64_t> reanalyzed{0};
};

struct SelfPlayConfig {
  int   games_per_actor = 64;
  // playout cap randomization (KataGo): a move gets a full search with
  // probability full_search_prob and becomes a policy training target,
  // other moves get a fast search that only advances the game; values are
  // trained on every move. Set full_search_prob = 1 to disable.
  int   num_simulations = 64;
  int   fast_simulations = 16;
  float full_search_prob = 0.4f;
  // visit count sampling for searches that use it (PUCT)
  float temperature = 1.f;
  int   temperature_moves = 20;
  // reanalyze: positions refreshed per sampled trajectory
  int   reanalyze_positions = 16;
};

// a worker thread playing games_per_actor games at once against itself;
// every step it advances each game's search to its next network request,
// sends all of them in one inference call, then feeds the results back
template <MuZeroGame Game>
struct SelfPlayActor {
  SelfPlayActor(
      const SelfPlayConfig& config,
      const Encoder<Game>* encoder,
      Inference* inference,
      SearchFactory<Game> make_search,
      ReplayBuffer* replay,
      ExperienceStore* store,
      SelfPlayStats* stats,
      s::function<uint32_t()> model_version,
      uint64_t seed)
  : mConfig(config), mEncoder(encoder), mInference(inference), mReplay(replay),
    mStore(store), mStats(stats), mModelVersion(s::move(model_version)), mRecorder(encoder), mRng(seed)
  {
    for (int i = 0; i < config.games_per_actor; ++i)
      mSlots.push_back(Slot{Game(), make_search(mRng()), nullptr, false, 0});
  }

  // pause, when set, is polled every step; self-play waits while it returns
  // true, e.g. while the learner lags behind the data
  void set_pause(s::function<bool()> pause){ mPause = s::move(pause); }

  void run(const s::atomic<bool>& stop){
    InferenceRequests requests;
    requests.observation_size = mInference->observation_size();
    requests.latent_size = mInference->latent_size();
    InferenceResults results;
    s::vector<int> waiting;
    for (Slot& slot : mSlots) new_game(slot);
    while (not stop.load(s::memory_order_relaxed)){
      if (mPause and mPause()){
        s::this_thread::sleep_for(s::chrono::milliseconds(5));
        continue;
      }
      requests.clear();
      waiting.clear();
      for (int i = 0; i < (int)mSlots.size(); ++i){
        Slot& slot = mSlots[i];
        while (not slot.search->step(requests)) play_move(slot);
        waiting.push_back(i);
      }
      mInference->infer(requests, results);
      for (int i : waiting) mSlots[i].search->receive(results);
    }
  }

protected:
  struct Slot {
    Game                      game;
    s::unique_ptr<Search<Game>> search;
    s::shared_ptr<Trajectory> trajectory;
    bool                      full;
    int                       move;
  };

  void new_game(Slot& slot){
    slot.game = Game();
    slot.trajectory = mRecorder.begin(mModelVersion());
    slot.move = 0;
    start_search(slot);
  }

  void start_search(Slot& slot){
    s::uniform_real_distribution<float> u(0.f, 1.f);
    slot.full = u(mRng) < mConfig.full_search_prob;
    slot.search->start(slot.game, slot.full ? mConfig.num_simulations : mConfig.fast_simulations, true);
  }

  void play_move(Slot& slot){
    float temperature = slot.move < mConfig.temperature_moves ? mConfig.temperature : 0.f;
    SearchResult result = slot.search->result(temperature);
    mRecorder.move(*slot.trajectory, slot.game, result, slot.full or result.simulations == 0);
    ++slot.move;
    mStats->moves += 1;
    if (slot.full) mStats->full_searches += 1;
    if (slot.game.is_terminal()){
      mRecorder.finish(*slot.trajectory, slot.game);
      finish_game(slot.trajectory);
      new_game(slot);
    } else {
      start_search(slot);
    }
  }

  void finish_game(s::shared_ptr<Trajectory> t){
    if (mStore) mStore->write(*t);
    mStats->positions += t->length;
    mStats->games += 1;
    if (t->outcome > 0)      mStats->wins += 1;
    else if (t->outcome < 0) mStats->losses += 1;
    else                     mStats->draws += 1;
    mReplay->add(s::move(t));
  }

  SelfPlayConfig           mConfig;
  const Encoder<Game>*     mEncoder;
  Inference*               mInference;
  ReplayBuffer*            mReplay;
  ExperienceStore*         mStore;
  SelfPlayStats*           mStats;
  s::function<uint32_t()>  mModelVersion;
  TrajectoryRecorder<Game> mRecorder;
  s::mt19937_64            mRng;
  s::vector<Slot>          mSlots;
  s::function<bool()>      mPause;
};

// MuZero Reanalyse: re-runs full searches with the latest network on
// positions already in the replay buffer and replaces their policy targets
// and root values, so old games keep yielding targets as good as the current
// network can make. The root values, tagged with model_version, are the
// fresh bootstrap values of EfficientZero's off-policy value correction.
template <MuZeroGame Game>
struct ReanalyzeActor {
  ReanalyzeActor(
      const SelfPlayConfig& config,
      Inference* inference,
      SearchFactory<Game> make_search,
      ReplayBuffer* replay,
      SelfPlayStats* stats,
      s::function<uint32_t()> model_version,
      uint64_t seed)
  : mConfig(config), mInference(inference), mReplay(replay), mStats(stats),
    mModelVersion(s::move(model_version)), mRng(seed)
  {
    for (int i = 0; i < config.games_per_actor; ++i)
      mSlots.push_back(Slot{Game(), make_search(mRng()), nullptr, 0, 0});
  }

  void run(const s::atomic<bool>& stop){
    InferenceRequests requests;
    requests.observation_size = mInference->observation_size();
    requests.latent_size = mInference->latent_size();
    InferenceResults results;
    s::vector<int> waiting;
    for (Slot& slot : mSlots) next_trajectory(slot);
    while (not stop.load(s::memory_order_relaxed)){
      requests.clear();
      waiting.clear();
      for (int i = 0; i < (int)mSlots.size(); ++i){
        Slot& slot = mSlots[i];
        if (not slot.trajectory) next_trajectory(slot);
        if (not slot.trajectory) continue;
        bool waits = false;
        while (slot.trajectory and not (waits = slot.search->step(requests))) advance(slot);
        if (waits) waiting.push_back(i);
      }
      if (waiting.empty()){
        // replay buffer still empty
        s::this_thread::sleep_for(s::chrono::milliseconds(100));
        continue;
      }
      mInference->infer(requests, results);
      for (int i : waiting) mSlots[i].search->receive(results);
    }
  }

protected:
  struct Slot {
    Game                      game;
    s::unique_ptr<Search<Game>> search;
    s::shared_ptr<Trajectory> trajectory;
    int                       t;
    int                       end;
  };

  void next_trajectory(Slot& slot){
    slot.trajectory = mReplay->sample_trajectory(mRng);
    if (not slot.trajectory) return;
    const Trajectory& tr = *slot.trajectory;
    s::uniform_int_distribution<int> start(0, tr.length - 1);
    slot.t = start(mRng);
    slot.end = s::min(tr.length, slot.t + mConfig.reanalyze_positions);
    slot.game = Game();
    for (int i = 0; i < slot.t; ++i) slot.game.apply(tr.actions[i]);
    slot.search->start(slot.game, mConfig.num_simulations, true);
  }

  void advance(Slot& slot){
    Trajectory& tr = *slot.trajectory;
    SearchResult result = slot.search->result(0.f);
    {
      s::lock_guard<s::mutex> lock(tr.policy_mutex);
      s::copy(result.policy.begin(), result.policy.end(), tr.policies.begin() + (size_t)slot.t * tr.num_actions);
      tr.policy_valid[slot.t] = 1;
      if (result.simulations > 0){
        tr.root_values[slot.t] = result.root_value;
        tr.value_versions[slot.t] = mModelVersion();
      }
    }
    mStats->reanalyzed += 1;
    slot.game.apply(tr.actions[slot.t]);
    ++slot.t;
    if (slot.t >= slot.end){
      tr.reanalyzed += 1;
      next_trajectory(slot);
    } else {
      slot.search->start(slot.game, mConfig.num_simulations, true);
    }
  }

  SelfPlayConfig          mConfig;
  Inference*              mInference;
  ReplayBuffer*           mReplay;
  SelfPlayStats*          mStats;
  s::function<uint32_t()> mModelVersion;
  s::mt19937_64           mRng;
  s::vector<Slot>         mSlots;
};

}  // namespace rlbg

#endif//RLBG_SELFPLAY_ACTOR_H
