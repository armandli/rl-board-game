#ifndef RLBG_EXPERIENCE_REPLAY_BUFFER_H
#define RLBG_EXPERIENCE_REPLAY_BUFFER_H

#include <cstdint>

#include <memory>
#include <random>
#include <vector>

#include <experience/trajectory.h>

namespace rlbg {

namespace s = std;

struct PositionRef {
  s::shared_ptr<const Trajectory> trajectory;
  int                             t;
};

// holds recent trajectories for training; thread safe
struct ReplayBuffer {
  virtual ~ReplayBuffer() = default;

  virtual void add(s::shared_ptr<Trajectory> trajectory) = 0;
  // non-terminal positions
  virtual void sample_positions(int n, s::mt19937_64& rng, s::vector<PositionRef>& out) const = 0;
  virtual s::shared_ptr<Trajectory> sample_trajectory(s::mt19937_64& rng) const = 0;

  // positions and games currently held
  virtual size_t num_positions() const = 0;
  virtual size_t num_games() const = 0;
  // positions and games ever added
  virtual uint64_t total_positions() const = 0;
  virtual uint64_t total_games() const = 0;
};

}  // namespace rlbg

#endif//RLBG_EXPERIENCE_REPLAY_BUFFER_H
