#ifndef RLBG_MLX_UTIL_NN_H
#define RLBG_MLX_UTIL_NN_H

// C++ counterpart of Python's mlx.nn and mlx.optimizers:
//   rlbg::nn         Module, layers, activations, losses, value_and_grad,
//                    checkpoint, weight file io
//   rlbg::nn::init   initializers
//   rlbg::optim      optimizers, schedulers, clip_grad_norm

#include <mlx_util/activations.h>
#include <mlx_util/containers.h>
#include <mlx_util/convolution.h>
#include <mlx_util/dropout.h>
#include <mlx_util/embedding.h>
#include <mlx_util/init.h>
#include <mlx_util/io.h>
#include <mlx_util/linear.h>
#include <mlx_util/losses.h>
#include <mlx_util/module.h>
#include <mlx_util/normalization.h>
#include <mlx_util/optimizers.h>
#include <mlx_util/pooling.h>
#include <mlx_util/positional_encoding.h>
#include <mlx_util/recurrent.h>
#include <mlx_util/schedulers.h>
#include <mlx_util/transformer.h>
#include <mlx_util/upsample.h>

#endif//RLBG_MLX_UTIL_NN_H
