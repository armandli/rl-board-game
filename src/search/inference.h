#ifndef RLBG_SEARCH_INFERENCE_H
#define RLBG_SEARCH_INFERENCE_H

#include <cstdint>
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif
#include <string>
#include <vector>

namespace rlbg {

namespace s = std;

// latents travel between the network and the search tree in half precision,
// which halves the tree's memory and the transfer to the device. IEEE binary16,
// the same bits as mlx::core::float16_t and c10::Half, so backends memcpy it.
// libc++ has no <stdfloat> yet; clang's builtin _Float16 is the same type
#if defined(__STDCPP_FLOAT16_T__)
using latent_t = s::float16_t;
#else
using latent_t = _Float16;
#endif

// a batch of network evaluations requested by searches: initial inference
// (represent + predict) of observations and recurrent inference (dynamics +
// predict) of latent/action pairs
struct InferenceRequests {
  int                   observation_size = 0;
  int                   latent_size = 0;
  s::vector<uint8_t>    features;   // num_initial() * observation_size
  s::vector<latent_t>   latents;    // num_recurrent() * latent_size
  s::vector<int32_t>    actions;    // num_recurrent()

  int num_initial() const { return observation_size ? (int)(features.size() / observation_size) : 0; }
  int num_recurrent() const { return (int)actions.size(); }
  int size() const { return num_initial() + num_recurrent(); }
  void clear(){
    features.clear();
    latents.clear();
    actions.clear();
  }

  // these return the request index within its kind
  int add_initial(const uint8_t* obs){
    int i = num_initial();
    features.insert(features.end(), obs, obs + observation_size);
    return i;
  }
  int add_recurrent(const latent_t* latent, int action){
    int i = num_recurrent();
    latents.insert(latents.end(), latent, latent + latent_size);
    actions.push_back(action);
    return i;
  }
};

// results in request order: every initial request, then every recurrent one
struct InferenceResults {
  int                 num_initial = 0;
  int                 latent_size = 0;
  int                 num_actions = 0;
  s::vector<latent_t> latents;        // size * latent_size
  s::vector<float>    policy_logits;  // size * num_actions
  s::vector<float>    values;         // size, mover's perspective in [-1, 1]
  s::vector<float>    rewards;        // size, 0 for initial inference

  int initial_index(int i) const { return i; }
  int recurrent_index(int i) const { return num_initial + i; }
  const latent_t* latent(int idx) const { return latents.data() + (size_t)idx * latent_size; }
  const float* logits(int idx) const { return policy_logits.data() + (size_t)idx * num_actions; }
};

// evaluates requests with some network; implementations may batch requests
// of many callers together, so infer() must be thread safe
struct Inference {
  virtual ~Inference() = default;
  virtual void infer(const InferenceRequests& requests, InferenceResults& results) = 0;
  virtual int observation_size() const = 0;
  virtual int latent_size() const = 0;
  virtual int num_actions() const = 0;
};

// an Inference whose network weights load from a snapshot file written by
// the same ML framework; evaluation and play load snapshots through it
struct SnapshotInference : Inference {
  virtual void load_weights(const s::string& file) = 0;
};

}  // namespace rlbg

#endif//RLBG_SEARCH_INFERENCE_H
