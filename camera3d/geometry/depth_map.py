from typing import Union, Literal
import cv2
from pathlib import Path
import numpy as np
import torch

from enum import Enum
from camera3d import Camera


class DepthMapType(Enum):
    Z_Depth = "z-depth"
    Distance = "distance"


class DepthMap:
    def __init__(
        self,
        depth_map: Union[np.ndarray, torch.Tensor],
        depth_type: DepthMapType = DepthMapType.Z_Depth,
        camera: Camera = None,
        device: str = "cpu",
    ):

        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)

        self.depth_type = depth_type
        self.depth_map = depth_map.to(torch.float32).to(device)
        self.valid_mask = self.depth_map >= torch.finfo(torch.float32).min
        self.camera = camera
        if self.camera is not None:
            self.camera = self.camera.to(device)

    def get_pointcloud(self):
        assert self.camera is not None
        directions = self.camera.pixelwise_directions().squeeze(0)  # [h, w, 3]
        print(directions.shape, self.depth_map.shape)
        if self.depth_type == DepthMapType.Z_Depth:
            xyz = directions / directions[..., 2:] * self.depth_map.unsqueeze(-1)
        elif self.depth_type == DepthMapType.Distance:
            xyz = directions / torch.linalg.norm(directions, dim=-1) * self.depth_map.unsqueeze(-1)
        else:
            raise NotImplementedError(f"unknown depth map type:{self.depth_type}")
        return xyz

    def get_depth(self, depth_type: DepthMapType) -> torch.Tensor:
        if self.depth_type == depth_type:
            return self.depth_map
        elif self.depth_type == DepthMapType.Z_Depth and depth_type == DepthMapType.Distance:
            # zdepth to distance
            return self.zdepth_to_distance()
        elif self.depth_type == DepthMapType.Distance and depth_type == DepthMapType.Z_Depth:
            return self.distance_to_zdepth()
        else:
            raise NotImplementedError(f"unknown type: {self.depth_type} and {depth_type}")

    def zdepth_to_distance(self) -> torch.Tensor:
        xyz = self.get_pointcloud()
        distance = torch.linalg.norm(xyz, dim=-1)
        distance[~self.valid_mask] = 0.0
        return distance

    def distance_to_zdepth(self) -> torch.Tensor:
        xyz = self.get_pointcloud()
        zdepth = xyz[..., 2]
        zdepth[~self.valid_mask] = 0.0
        return zdepth

    def get_ray_near_far(self, sigma: Union[float, torch.Tensor]):
        """
        Gaussian distribution: N(depth, sigma) for per-pixel depth
        near_far = [depth-3sigma, depth+3sigma]
        """
        distance_map = self.get_depth(DepthMapType.Distance)
        depth_near = distance_map - 3 * sigma
        depth_far = distance_map + 3 * sigma
        near_far = torch.stack([depth_near, depth_far], dim=-1)
        return near_far

    def get_color_map(self):
        """return color map for per-pixel depth"""
        valid_depth = np.full_like(self.depth_map, 0)
        valid_depth[self.valid_mask] = self.depth_map[self.valid_mask]
        valid_depth = cv2.normalize(valid_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(valid_depth, cv2.COLORMAP_JET)
        return depth_color

    @staticmethod
    def load_depth(depth_path: Path, height=None, width=None) -> np.ndarray:
        if not Path(depth_path).exists():
            print(f"{depth_path} does not exist. All depth values are set to be invalid.")
            depth = np.zeros((height, width), dtype=np.float32)
        else:
            with open(depth_path, "rb") as f:
                w, h = np.fromfile(f, dtype=np.longlong, count=2)
                depth_min, depth_max = np.fromfile(f, dtype=np.float32, count=2)
                depth = np.fromfile(f, dtype=np.uint16, count=w * h).reshape(h, w)
                depth = depth.astype(np.float32)
                depth = depth / np.iinfo(np.uint16).max * (depth_max - depth_min) + depth_min
            if (height is not None and width is not None) and (w != width or h != height):
                depth = cv2.resize(depth, [width, height], interpolation=cv2.INTER_NEAREST)
        return DepthMap(depth, DepthMapType.Z_Depth)

    def save_depth(self, depth_path: str) -> None:
        depth = self.depth_map.cpu().numpy()
        depth_shape = np.array(depth.shape[::-1], dtype=np.longlong)
        depth_min = depth.min()
        depth_max = depth.max()

        depth_min_max = np.array((depth_min, depth_max), dtype=np.float32)
        depth_uint16 = (depth - depth_min) * np.iinfo(np.uint16).max / (depth_max - depth_min)
        depth_uint16 = np.round(depth_uint16).astype(np.uint16)
        with open(depth_path, "wb") as outfile:
            outfile.write(depth_shape.tobytes())
            outfile.write(depth_min_max.tobytes())
            outfile.write(depth_uint16.tobytes())

    def set_camera(self, camera: Camera):
        self.camera = camera.to(self.depth_map.device)



def filter_depth_map(depth: np.ndarray):
    valid_mask = depth > 0.
    valid_values = depth[valid_mask]

    Q3 = np.percentile(valid_values, 75)
    Q1 = np.percentile(valid_values, 25)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR

    valid_mask = valid_mask & (depth < max_value*2)

    depth = depth.copy()
    depth[~valid_mask] = 0.
    return depth