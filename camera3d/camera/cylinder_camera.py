import torch
from .base_camera import Camera, CameraModel
import math


class CylinderCamera(Camera):

    # hfov
    model = CameraModel.Cylinder

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:

        hfov = self.params.squeeze()
        h, w = self.hws.unbind(dim=-1)

        f = w / hfov

        cx = w * 0.5
        cy = h * 0.5

        u, v = uv.unbind(dim=-1)

        lon = (u - cx) / f
        lat = torch.atan2(v - cy, f)

        cos_lat = torch.cos(lat)
        x = cos_lat * torch.sin(lon)
        y = torch.sin(lat)
        z = cos_lat * torch.cos(lon)
        return torch.stack([x, y, z], dim=-1)

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        # assert torch.equal(self.hws[..., 1], self.hws[..., 0] * 2.0)

        hfov = self.params.squeeze()

        x, y, z = points.unbind(dim=-1)
        lon = torch.atan2(x, z)
        lat = torch.atan2(y, torch.hypot(x, z))

        h, w = self.hws.unbind(dim=-1)
        f = w / hfov

        cx = w * 0.5
        cy = h * 0.5

        u = lon * f + cx
        v = torch.tan(lat) * f + cy
        return torch.stack([u, v], dim=-1)
