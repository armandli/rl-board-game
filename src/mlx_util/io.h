#ifndef RLBG_MLX_UTIL_IO_H
#define RLBG_MLX_UTIL_IO_H

#include <string>
#include <unordered_map>

#include <mlx/mlx.h>

#include <mlx_util/module.h>

namespace rlbg::nn {

namespace s = std;
namespace mx = mlx::core;

// counterparts of Python's mx.load / mx.savez / mx.savez_compressed /
// mx.save_safetensors for dictionaries of arrays; files are interchangeable
// with Python MLX and numpy (.npz)

// dispatches on extension: .npz, .safetensors or .gguf
Params load_arrays(const s::string& file);
// dispatches on extension: .npz or .safetensors
void save_arrays(const s::string& file, const Params& arrays);

Params load_npz(const s::string& file);
// compressed uses deflate like mx.savez_compressed; needs zlib
void save_npz(const s::string& file, const Params& arrays, bool compressed = false);

Params load_safetensors(const s::string& file);
void save_safetensors(
  const s::string& file,
  const Params& arrays,
  const s::unordered_map<s::string, s::string>& metadata = {}
);

bool has_npz_compression();

} // namespace rlbg::nn

#endif//RLBG_MLX_UTIL_IO_H
