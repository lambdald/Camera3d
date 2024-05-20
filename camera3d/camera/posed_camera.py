from .base_camera import Camera, CameraModel
from .pinhole_camera import PinholeCamera
import torch
from camera3d.geometry.ray import Ray
from camera3d.geometry.transform import CoordinateType, Transform3d
import numpy as np
import math


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.eye(4, device=R.device)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W).float()
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class PosedCamera:
    def __init__(self, cameras: Camera, poses_c2w: Transform3d) -> None:
        self.cameras = cameras
        self.poses_w2c = poses_c2w


    def transorm_to_coordinates(self, directions: torch.Tensor, coord_type: CoordinateType):
        if coord_type == CoordinateType.OpenGL:
            directions = directions * torch.tensor([1, -1, -1], device=directions.device)
        elif coord_type == CoordinateType.OpenCV:
            # camera is in opencv coordiante.
            pass
        else:
            raise NotImplementedError(f"unknown coorinate type:{coord_type}")
        return directions

    def get_directions(self, uv: torch.Tensor) -> torch.Tensor:
        device = uv.device
        cameras: Camera = self.cameras.to(device)
        directions = cameras.backproject_to_3d(uv)
        return self.transorm_to_coordinates(directions, self.poses_w2c.coord_type)

    def get_pixelwise_rays(self):

        pose_c2w = self.poses_w2c.transforms
        coord_type = self.poses_w2c.coord_type

        cam: Camera = self.cameras.to(pose_c2w.device)
        directions = cam.pixelwise_directions()

        directions = self.transorm_to_coordinates(directions, coord_type)
        ray_dir = (pose_c2w[..., :3, :3] @ directions.unsqueeze(-1)).squeeze(-1)
        ray_origin = pose_c2w[..., :3, 3].expand_as(ray_dir)

        ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

        return Ray(origin=ray_origin, direction=ray_dir, batch_size=ray_origin.shape[:-1])

    def get_rays(self, uv: torch.Tensor) -> Ray:
        pose_c2w = torch.inverse(self.poses_w2c.transforms)
        directions = self.get_directions(uv)
        ray_dir = (pose_c2w[..., :3, :3] @ directions.unsqueeze(-1)).squeeze(-1)
        ray_origin = pose_c2w[..., :3, 3]
        ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
        return Ray(origin=ray_origin, direction=ray_dir, batch_size=uv.shape[:-1])


    def get_transforms(self, device, near: float, far: float):
        """
        Get MVP for 3d gaussian splatting.
        The pose should be opencv coordinate(colmap coordinate).
        ref: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/cameras.py
        """
        # because the specific matrix storage of glm library, the transform may by confused.
        # todo: change glm to eigen
        assert Transform3d.coord_type == CoordinateType.OpenCV

        pose_w2c = self.poses_w2c.transforms.to(device)

        assert (
            self.cameras.model == CameraModel.Pinhole or self.cameras.model == CameraModel.FoV
        ), "Only support pinhole and fov camera."

        if self.cameras.model == CameraModel.FoV:
            camera: PinholeCamera = self.cameras.pinhole().to(device)
        elif self.cameras.model == CameraModel.Pinhole:
            camera: PinholeCamera = self.cameras.to(device)
        else:
            raise ValueError("Only support pinhole and fov camera.")


        projection_matrix = camera.projection_matrix(near=near, far=far).squeeze().transpose(0, 1)

        R = pose_w2c.squeeze(0)[:3, :3]
        T = pose_w2c.squeeze(0)[:3, 3]

        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        return world_view_transform, projection_matrix, full_proj_transform
