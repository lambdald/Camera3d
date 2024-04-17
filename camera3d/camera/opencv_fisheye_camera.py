import torch
from .base_camera import Camera, CameraModel
from camera3d.cpp_backend import cpp_backend


class OpenCVFisheyeCamera(Camera):

    # fx, fy, cx, cy, k1, k2, k3, k4
    model = CameraModel.OpenCVFisheye

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        prefix = uv.shape[:-1]
        f = self.params[..., :2].squeeze()
        c = self.params[..., 2:4].squeeze()
        d = self.params[..., 4:8].squeeze()

        xy = (uv - c) / f
        xy = cpp_backend.Undistort(cpp_backend.CameraModel.OpenCVFisheye, xy.view(-1, 2), d.view(-1, 4).squeeze(0)).view(prefix+(2,))
        z = torch.ones(xy.shape[:-1], dtype=torch.float32, device=uv.device).unsqueeze(-1)

        xyz = torch.cat([xy, z], dim=-1)
        return xyz

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        prefix = points.shape[:-1]

        f = self.params[..., :2].squeeze()
        c = self.params[..., 2:4].squeeze()
        d = self.params[..., 4:8].squeeze()
        xy = points[..., :2] / points[..., 2:]
        xy = cpp_backend.Distort(cpp_backend.CameraModel.OpenCVFisheye, xy.view(-1, 2), d.view(-1, 4).squeeze(0)).view(prefix+(2,))
        uv = f * xy + c
        return uv
