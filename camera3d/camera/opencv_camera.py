import torch
from .base_camera import Camera, CameraModel
from camera3d.cpp_backend import cpp_backend


class OpenCVCamera(Camera):

    # fx, fy, cx, cy, k1, k2, p1, p2, k3
    # refer to https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    model = CameraModel.OpenCV

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        f = self.params[..., :2]
        c = self.params[..., 2:4]
        d = self.params[..., 4:9]

        xy = (uv - c) / f

        xy = cpp_backend.Undistort(cpp_backend.CameraModel.OpenCV, xy, d)

        z = torch.ones(self.batch_size, dtype=torch.float32, device=uv.device).unsqueeze(-1)

        xyz = torch.cat([xy, z], dim=-1)
        return xyz

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        f = self.params[..., :2]
        c = self.params[..., 2:4]
        d = self.params[..., 4:9]
        xy = points[..., :2] / points[..., 2:]
        xy = cpp_backend.Distort(cpp_backend.CameraModel.OpenCV, xy, d)
        uv = f * xy + c
        return uv
