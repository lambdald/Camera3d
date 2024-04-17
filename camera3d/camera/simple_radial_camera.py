import torch
from .base_camera import Camera, CameraModel
from camera3d.cpp_backend import cpp_backend


class SimpleRadialCamera(Camera):

    # f, cx, cy, k
    model: CameraModel = CameraModel.SimpleRadial

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        prefix = uv.shape[:-1]
        f = self.params[..., 0:1].squeeze().unsqueeze(-1)
        c = self.params[..., 1:3].squeeze()
        d = self.params[..., 3:4].squeeze()
        xy = (uv - c) / f
        xy = cpp_backend.Undistort(cpp_backend.CameraModel.SimpleRadial, xy.view(-1, 2), d.view(-1, 1).squeeze(0)).view(prefix+(2,))
        z = torch.ones(xy.shape[:-1], dtype=torch.float32, device=uv.device).unsqueeze(-1)
        xyz = torch.cat([xy, z], dim=-1)
        return xyz

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        prefix = points.shape[:-1]

        f = self.params[..., 0:1].squeeze().unsqueeze(-1)
        c = self.params[..., 1:3].squeeze()
        d = self.params[..., 3:4].squeeze()
        xy = points[..., :2] / points[..., 2:]
        xy = cpp_backend.Distort(cpp_backend.CameraModel.SimpleRadial, xy.view(-1, 2), d.view(-1, 1).squeeze(0)).view(prefix+(2,))
        uv = f * xy + c
        return uv
