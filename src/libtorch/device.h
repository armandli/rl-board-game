#ifndef RLBG_LIBTORCH_DEVICE_H
#define RLBG_LIBTORCH_DEVICE_H

#include <optional>
#include <string>

#include <torch/torch.h>

#ifdef RLBG_TORCH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace rlbg::libtorch {

namespace s = std;
namespace t = torch;

// "auto" picks CUDA when available, otherwise "cuda" or "cpu"
inline t::Device parse_device(const s::string& name){
  if (name == "cpu") return t::kCPU;
  if (name == "cuda" or name == "gpu" or (name == "auto" and t::cuda::is_available())) return t::kCUDA;
  return t::kCPU;
}

// a CUDA stream of its own for one component (learner, inference), so their
// kernels overlap on the GPU; a no-op on the CPU. Work runs on it inside a
// Scope.
struct DeviceStream {
  explicit DeviceStream(t::Device device){
#ifdef RLBG_TORCH_CUDA
    if (device.is_cuda()) mStream = c10::cuda::getStreamFromPool(false, device.index() < 0 ? 0 : device.index());
#endif
  }

  struct Scope {
#ifdef RLBG_TORCH_CUDA
    explicit Scope(const DeviceStream& ds){
      if (ds.mStream) mGuard.emplace(*ds.mStream);
    }
    s::optional<c10::cuda::CUDAStreamGuard> mGuard;
#else
    explicit Scope(const DeviceStream&){}
#endif
  };

  // waits until all work queued on the stream is done
  void synchronize() const {
#ifdef RLBG_TORCH_CUDA
    if (mStream) mStream->synchronize();
#endif
  }

  bool pinned() const {
#ifdef RLBG_TORCH_CUDA
    return mStream.has_value();
#else
    return false;
#endif
  }

#ifdef RLBG_TORCH_CUDA
  s::optional<c10::cuda::CUDAStream> mStream;
#endif
};

}  // namespace rlbg::libtorch

#endif//RLBG_LIBTORCH_DEVICE_H
