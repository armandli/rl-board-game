#ifndef RLBG_EVALUATION_EVALUATOR_H
#define RLBG_EVALUATION_EVALUATOR_H

#include <atomic>
#include <string>

namespace rlbg {

namespace s = std;

// measures whether training makes progress by playing model snapshots
// against each other and against fixed baselines
struct Evaluator {
  virtual ~Evaluator() = default;

  // queues a snapshot saved as weights_file under the given player name
  virtual void submit(const s::string& name, const s::string& weights_file) = 0;
  // evaluates queued snapshots until stop is set
  virtual void run(const s::atomic<bool>& stop) = 0;
  // one line describing the latest evaluation
  virtual s::string summary() const = 0;
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_EVALUATOR_H
