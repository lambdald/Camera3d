#pragma once

#include <torch/torch.h>


#define CUDA_CHECK(stmt)                                                                                 \
  do {                                                                                                   \
    cudaError_t result = (stmt);                                                                         \
    if (cudaSuccess != result) {                                                                         \
      fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result)); \
      exit(-1);                                                                                          \
    }                                                                                                    \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_KERNEL_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CUDA_GET_THREAD_ID(tid, Q)                       \
  const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define DEVICE_GUARD(_ten) const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));
