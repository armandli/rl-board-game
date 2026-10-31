#ifndef RLBG_EXPERIENCE_FIFO_REPLAY_BUFFER_H
#define RLBG_EXPERIENCE_FIFO_REPLAY_BUFFER_H

#include <cstdint>

#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>

#include <experience/replay_buffer.h>
#include <experience/trajectory.h>

namespace rlbg {

namespace s = std;

// sliding window over the most recent `capacity` positions with uniform
// sampling over positions, as MuZero does for board games. Each trajectory
// records its first position's global index, so sampling a position is a
// binary search over the window.
struct FifoReplayBuffer : ReplayBuffer {
  explicit FifoReplayBuffer(size_t capacity): mCapacity(capacity) {}

  void add(s::shared_ptr<Trajectory> trajectory) override {
    s::unique_lock lock(mMutex);
    trajectory->id = mTotalGames;
    mEntries.push_back(Entry{mTotalPositions, s::move(trajectory)});
    mTotalPositions += mEntries.back().trajectory->length;
    mPositions += mEntries.back().trajectory->length;
    ++mTotalGames;
    while (mEntries.size() > 1 and mPositions - mEntries.front().trajectory->length >= mCapacity){
      mPositions -= mEntries.front().trajectory->length;
      mEntries.pop_front();
    }
  }

  void sample_positions(int n, s::mt19937_64& rng, s::vector<PositionRef>& out) const override {
    s::shared_lock lock(mMutex);
    out.clear();
    if (mEntries.empty()) return;
    uint64_t lo = mEntries.front().start;
    uint64_t hi = mEntries.back().start + mEntries.back().trajectory->length;
    s::uniform_int_distribution<uint64_t> dist(lo, hi - 1);
    for (int i = 0; i < n; ++i){
      uint64_t g = dist(rng);
      auto it = s::upper_bound(mEntries.begin(), mEntries.end(), g,
        [](uint64_t v, const Entry& e){ return v < e.start; });
      --it;
      out.push_back(PositionRef{it->trajectory, (int)(g - it->start)});
    }
  }

  s::shared_ptr<Trajectory> sample_trajectory(s::mt19937_64& rng) const override {
    s::shared_lock lock(mMutex);
    if (mEntries.empty()) return nullptr;
    s::uniform_int_distribution<size_t> dist(0, mEntries.size() - 1);
    return mEntries[dist(rng)].trajectory;
  }

  size_t num_positions() const override {
    s::shared_lock lock(mMutex);
    return mPositions;
  }
  size_t num_games() const override {
    s::shared_lock lock(mMutex);
    return mEntries.size();
  }
  uint64_t total_positions() const override {
    s::shared_lock lock(mMutex);
    return mTotalPositions;
  }
  uint64_t total_games() const override {
    s::shared_lock lock(mMutex);
    return mTotalGames;
  }

protected:
  struct Entry {
    uint64_t                  start;
    s::shared_ptr<Trajectory> trajectory;
  };

  size_t                  mCapacity;
  mutable s::shared_mutex mMutex;
  s::deque<Entry>         mEntries;
  size_t                  mPositions = 0;
  uint64_t                mTotalPositions = 0;
  uint64_t                mTotalGames = 0;
};

}  // namespace rlbg

#endif//RLBG_EXPERIENCE_FIFO_REPLAY_BUFFER_H
