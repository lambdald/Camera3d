#include <torch/torch.h>

#include <Eigen/Eigen>

#include "Camera.h"
#include "cuda.cuh"
#include "types.h"

using Tensor = torch::Tensor;

__device__ void SimpleRadialDistort(const float* distortion_params, const float u, const float v, float* du, float* dv) {
  const auto k = distortion_params[0];

  const auto u2 = u * u;
  const auto v2 = v * v;
  const auto r2 = u2 + v2;
  const auto radial = k * r2;
  *du = u * radial;
  *dv = v * radial;
}

__device__ __host__ void FisheyeDistort(const float* distortion_params, const float u, const float v, float* du,
                                        float* dv) {
  auto k1 = distortion_params[0];
  auto k2 = distortion_params[1];
  auto k3 = distortion_params[2];
  auto k4 = distortion_params[3];

  auto u2 = u * u;
  auto v2 = v * v;
  auto r2 = u2 + v2;

  auto theta = atan(r2);
  auto theta2 = theta * theta;
  auto theta4 = theta2 * theta2;
  auto theta6 = theta2 * theta4;
  auto theta8 = theta4 * theta4;

  auto theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

  *du = u * theta_d / r2;
  *dv = v * theta_d / r2;
}

__device__ __host__ void OpenCVDistort(const float* distortion_params, const float u, const float v, float* du, float* dv) {
  auto k1 = distortion_params[0];
  auto k2 = distortion_params[1];
  auto p1 = distortion_params[2];
  auto p2 = distortion_params[3];
  auto k3 = distortion_params[4];

  auto u2 = u * u;
  auto uv = u * v;
  auto v2 = v * v;
  auto r2 = u2 + v2;
  auto radial = k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;

  *du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2);
  *dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2);
}

// define static pointer to distort function in device
__device__ DistortionFunctor StaticSimpleRadialDistort = SimpleRadialDistort;
__device__ DistortionFunctor StaticFisheyeDistort = FisheyeDistort;
__device__ DistortionFunctor StaticOpenCVDistort = OpenCVDistort;

void GetDistortFunctor(DistortionFunctor& functor, CameraModel model) {
  if (model == CameraModel::SimpleRadial) {
    cudaMemcpyFromSymbol(&functor, StaticSimpleRadialDistort, sizeof(DistortionFunctor));
  } else if (model == CameraModel::OpenCV) {
    cudaMemcpyFromSymbol(&functor, StaticOpenCVDistort, sizeof(DistortionFunctor));
  } else if (model == CameraModel::OpenCVFisheye) {
    cudaMemcpyFromSymbol(&functor, StaticFisheyeDistort, sizeof(DistortionFunctor));
  } else {
    throw std::runtime_error("unknown camera model to distort" + std::to_string(static_cast<int>(model)));
  }
}

__device__ __host__ inline void _iterative_camera_undistortion(const float* params, Eigen::Vector2f* uv,
                                                               DistortionFunctor distort_func) {
  // Parameters for Newton iteration using numerical differentiation with
  // central differences, 100 iterations should be enough even for complex
  // camera models with higher order terms.
  const uint32_t kNumIterations = 100;
  const float kMaxStepNorm = 1e-10f;
  const float kRelStepSize = 1e-6f;

  Eigen::Matrix2f J;
  const Eigen::Vector2f x0 = *uv;
  Eigen::Vector2f x = *uv;
  Eigen::Vector2f dx;
  Eigen::Vector2f dx_0b;
  Eigen::Vector2f dx_0f;
  Eigen::Vector2f dx_1b;
  Eigen::Vector2f dx_1f;

  for (uint32_t i = 0; i < kNumIterations; ++i) {
    const float step0 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(0)));
    const float step1 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(1)));
    (*distort_func)(params, x(0), x(1), &dx(0), &dx(1));
    (*distort_func)(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    (*distort_func)(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    (*distort_func)(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    (*distort_func)(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
    J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
    const Eigen::Vector2f step_x = J.inverse() * (x + dx - x0);
    x -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }
  *uv = x;
}

__global__ void _CameraUndistortKernel(int n_pixels, const float* params, int n_params, Eigen::Vector2f* uv,
                                       DistortionFunctor distort_func) {
  int pix_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix_idx >= n_pixels) {
    return;
  }
  _iterative_camera_undistortion(params + pix_idx * n_params, uv + pix_idx, distort_func);
}

torch::Tensor Undistort(const CameraModel model, const torch::Tensor& uv, const torch::Tensor& dist_params) {
  int n_pixels = uv.sizes()[0];
  int n_params = dist_params.size(1);
  CHECK_EQ(n_pixels, dist_params.sizes()[0]);
  auto uv_cont = uv.contiguous();
  auto dist_params_cont = dist_params.contiguous();

  CHECK_KERNEL_INPUT(dist_params_cont);
  CHECK_KERNEL_INPUT(uv_cont);

  dim3 grid_dim = LIN_GRID_DIM(n_pixels);
  dim3 block_dim = LIN_BLOCK_DIM(n_pixels);

  DistortionFunctor hostFunc;
  GetDistortFunctor(hostFunc, model);

  _CameraUndistortKernel<<<grid_dim, block_dim>>>(n_pixels, dist_params_cont.data_ptr<float>(), n_params,
                                                  reinterpret_cast<Eigen::Vector2f*>(uv_cont.data_ptr()), hostFunc);

  return uv_cont;
}

__global__ void _CameraDistortKernel(int n_pixels, const float* params, int n_params, Eigen::Vector2f* uv,
                                     DistortionFunctor distort_func) {
  int pix_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix_idx >= n_pixels) {
    return;
  }

  const float* cur_params = params + pix_idx * n_params;
  Eigen::Vector2f* cur_uv = uv + pix_idx;
  Eigen::Vector2f new_uv(*cur_uv);
  distort_func(cur_params, (*cur_uv)(0), (*cur_uv)(1), &new_uv(0), &new_uv(1));
  *cur_uv += new_uv;
}

torch::Tensor Distort(const CameraModel model, const torch::Tensor& uv, const torch::Tensor& dist_params) {
  int n_pixels = uv.size(0);
  int n_params = dist_params.size(1);

  CHECK_EQ(n_pixels, dist_params.size(0));
  auto uv_cont = uv.contiguous();
  auto dist_params_cont = dist_params.contiguous();
  CK_CONT(dist_params_cont);
  CK_CONT(uv_cont);
  dim3 grid_dim = LIN_GRID_DIM(n_pixels);
  dim3 block_dim = LIN_BLOCK_DIM(n_pixels);
  DistortionFunctor hostFunc;
  GetDistortFunctor(hostFunc, model);
  _CameraDistortKernel<<<grid_dim, block_dim>>>(n_pixels, dist_params_cont.data_ptr<float>(), n_params,
                                                reinterpret_cast<Eigen::Vector2f*>(uv_cont.data_ptr()), hostFunc);
  return uv_cont;
}
