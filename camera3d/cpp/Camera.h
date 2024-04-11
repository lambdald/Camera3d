#pragma once

#include <torch/torch.h>
#include <functional>


enum class CameraModel {
  Unknown,
  Pinhole,
  FOV,
  SimpleRadial,
  OpenCV,
  OpenCVFisheye,
  Panoramic,
};


typedef void (*DistortionFunctor)(const float*, const float, const float, float*, float*);

torch::Tensor Undistort(const CameraModel model, const torch::Tensor& uv, const torch::Tensor& dist_params);
torch::Tensor Distort(const CameraModel model, const torch::Tensor& uv, const torch::Tensor& dist_params);
