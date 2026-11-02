#ifndef RLBG_LIBTORCH_MODEL_INFERENCE_H
#define RLBG_LIBTORCH_MODEL_INFERENCE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <torch/torch.h>

#include <search/inference.h>

#include <libtorch/device.h>
#include <libtorch/transformer_muzero.h>

namespace rlbg::libtorch {

namespace s = std;
namespace t = torch;

// "float32", "float16" or "bfloat16"
t::ScalarType parse_dtype(const s::string& name);

// runs a TransformerMuZero copy in reduced precision on its own CUDA stream,
// so inference overlaps the learner's work on the GPU. Calls are serialized
// by a mutex. Inputs and outputs go through pinned host buffers with one
// synchronization per call.
//
// Latents either cross to the host as raw IEEE half bits (latent_pool 0), or
// stay on the GPU in a ring buffer of latent_pool entries and the search
// holds a 64-bit handle in their place (latent_size() is then
// LATENT_HANDLE_SIZE). Searches treat latents as opaque, so both work with
// any Search; the pool removes 2 * T * D bytes of host copies and transfers
// per evaluation. A handle stays valid until latent_pool newer latents have
// been made; using an older one throws, so the pool must hold every latent of
// every search in flight.
struct ModelInference : SnapshotInference {
  static constexpr int LATENT_HANDLE_SIZE = sizeof(uint64_t) / sizeof(latent_t);

  ModelInference(const TransformerMuZeroConfig& config, t::Device device, t::ScalarType dtype,
                 int64_t latent_pool = 0);

  void infer(const InferenceRequests& requests, InferenceResults& results) override;
  int observation_size() const override { return (int)(mConfig.num_tokens * mConfig.num_features); }
  int latent_size() const override {
    return mPoolCapacity > 0 ? LATENT_HANDLE_SIZE : (int)(mConfig.num_tokens * mConfig.dims);
  }
  int num_actions() const override { return (int)mConfig.num_actions; }

  // a model file written by save_model with the same configuration
  void load_weights(const s::string& file) override;
  // copies src's weights (any device or dtype); waits until the copy is done
  void set_weights(TransformerMuZero& src);

  const TransformerMuZeroConfig& config() const { return mConfig; }
  t::Device device() const { return mDevice; }
  t::ScalarType dtype() const { return mDtype; }

protected:
  t::Tensor host_buffer(t::Tensor& buffer, int64_t numel, t::ScalarType type);
  // parents of recurrent requests: latents uploaded or looked up in the pool
  t::Tensor input_latents(const InferenceRequests& requests);
  // stores latent (n, T, D) and writes its n latents or handles to results
  void output_latents(const t::Tensor& latent, InferenceResults& results, bool async);

  TransformerMuZeroConfig mConfig;
  TransformerMuZero       mModel;
  t::Device               mDevice;
  t::ScalarType           mDtype;
  DeviceStream            mStream;
  s::mutex                mMutex;
  int64_t                 mPoolCapacity;
  t::Tensor               mPool;       // (capacity, T, D) on the device
  uint64_t                mNextHandle = 0;

  // pinned staging buffers, grown on demand
  t::Tensor               mFeatures, mLatentsIn, mActions, mSlots;
  t::Tensor               mLatentsOut, mScalarsOut;
};

}  // namespace rlbg::libtorch

#endif//RLBG_LIBTORCH_MODEL_INFERENCE_H
