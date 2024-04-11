import torch
from .base_camera import Camera, CameraModel, create_camera
import math


def fov2focal(fov, pixels):
    return pixels / (2 * torch.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * torch.atan(pixels / (2 * focal))


class FoVCamera(Camera):
    """
    For Graphic Rendering pipeline.
    """
    # fovx, fovy, cx, cy
    model = CameraModel.FoV

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        fov = self.params[..., :2]  # xfov yfov
        f = fov2focal(fov, torch.flip(self.hws, dims=[-1]))

        c = self.params[..., 2:]

        xy = (uv - c) / f

        z = torch.ones(self.batch_size, dtype=torch.float32, device=uv.device).unsqueeze(-1)

        xyz = torch.cat([xy, z], dim=-1)
        return xyz

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        fov = self.params[..., :2]
        f = fov2focal(fov, torch.flip(self.hws, dim=-1))

        c = self.params[..., 2:]
        xy = points[..., :2]
        z = points[..., 2:]
        uv = f * (xy / z) + c
        return uv

    def projection_matrix(self, near: float, far: float) -> torch.Tensor:
        # camera dose not know near and far of scene.
        xfov, yfov, cx, cy = torch.unbind(self.params, dim=-1)

        h, w = torch.unbind(self.hws, dim=-1)
        fx = fov2focal(xfov, w)
        fy = fov2focal(yfov, h)

        opengl_proj = torch.zeros(self.batch_size + (4, 4), dtype=torch.float32, device=self.device)

        opengl_proj[..., 0, 0] = 2 * fx / w
        opengl_proj[..., 0, 2] = -(w - 2 * cx) / w
        opengl_proj[..., 1, 1] = 2 * fy / h
        opengl_proj[..., 1, 2] = -(h - 2 * cy) / h
        opengl_proj[..., 2, 2] = far / (far - near)
        opengl_proj[..., 2, 3] = -(far * near) / (far - near)
        opengl_proj[..., 3, 2] = 1.0
        return opengl_proj

    def pinhole(self):
        fov = self.params[..., :2]  # xfov yfov
        f = fov2focal(fov, torch.flip(self.hws, dims=[-1]))

        params = self.params.clone()
        params[..., :2] = f
        return create_camera(CameraModel.Pinhole, self.hws, params)
