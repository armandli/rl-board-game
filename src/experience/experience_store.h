#ifndef RLBG_EXPERIENCE_EXPERIENCE_STORE_H
#define RLBG_EXPERIENCE_EXPERIENCE_STORE_H

#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include <experience/trajectory.h>

namespace rlbg {

namespace s = std;

// persists trajectories so a training run can resume with its replay buffer
// and rollouts can be inspected or reused offline; thread safe
struct ExperienceStore {
  virtual ~ExperienceStore() = default;

  virtual void write(const Trajectory& trajectory) = 0;
  virtual void flush() = 0;
  // most recent trajectories, oldest first, up to max_positions positions
  virtual s::vector<s::shared_ptr<Trajectory>> load_recent(size_t max_positions) = 0;
};

}  // namespace rlbg

#endif//RLBG_EXPERIENCE_EXPERIENCE_STORE_H
