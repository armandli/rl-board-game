#ifndef RLBG_SEARCH_SEARCH_H
#define RLBG_SEARCH_SEARCH_H

#include <vector>

#include <search/inference.h>

namespace rlbg {

namespace s = std;

struct SearchResult {
  int              action = -1;
  s::vector<float> policy;           // training target over all actions
  float            root_value = 0.f; // search value for the side to move
  int              simulations = 0;
};

// a planning search over one game position that runs as a resumable state
// machine, so a caller can drive many searches at once and evaluate all their
// network requests in one batch:
//
//   search.start(game, sims, noise);
//   while (search.step(requests)){  // one request added
//     inference.infer(requests, results);
//     search.receive(results);
//   }
//   SearchResult r = search.result(temperature);
template <typename Game>
struct Search {
  virtual ~Search() = default;

  // add_noise enables exploration noise for self-play
  virtual void start(const Game& root, int num_simulations, bool add_noise) = 0;
  // runs simulations until one needs a network evaluation, then adds that
  // request and returns true; returns false once the search is finished
  virtual bool step(InferenceRequests& requests) = 0;
  // delivers the evaluation of the request added by the last step()
  virtual void receive(const InferenceResults& results) = 0;
  virtual bool finished() const = 0;
  // temperature applies to searches that sample from visit counts
  virtual SearchResult result(float temperature) = 0;
};

}  // namespace rlbg

#endif//RLBG_SEARCH_SEARCH_H
