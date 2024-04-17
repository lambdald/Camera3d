from typing import Any
import torch
from enum import Enum

import torch
from tensordict import tensorclass
from enum import Enum
from dataclasses import dataclass
from typing import Union

from typing import Dict, Union
from abc import abstractmethod
import torch.nn.functional as F


@dataclass
class CameraAttribute:
    model_id: int
    model_name: str
    num_params: int

    def to_dict(self):
        data = dict(model_id=self.model_id, model_name=self.model_name, num_params=self.num_params)
        return data


class CameraModel(Enum):
    Unknown = CameraAttribute(model_id=0, model_name="unknown", num_params=0)
    Pinhole = CameraAttribute(model_id=1, model_name="pinhole", num_params=4)
    FoV = CameraAttribute(model_id=2, model_name="fov", num_params=4)
    SimpleRadial = CameraAttribute(model_id=3, model_name="simple_radial", num_params=4)
    OpenCV = CameraAttribute(model_id=4, model_name="opencv", num_params=9)
    OpenCVFisheye = CameraAttribute(model_id=5, model_name="opencv_fisheye", num_params=8)
    Panoramic = CameraAttribute(model_id=6, model_name="panoramic", num_params=0)


_camera_cls_ = {}


@tensorclass
class Camera:
    '''
    batch of camera
    '''
    hws: torch.Tensor
    '''shape: batchsize + [2]'''
    params: torch.Tensor
    '''shape: batchsize + [n]'''

    model: CameraModel = CameraModel.Unknown

    def __init_subclass__(cls) -> None:
        """
        In order to implement the camera factory method
        """
        _camera_cls_[cls.model.value.model_id] = cls
        _camera_cls_[cls.model.value.model_name] = cls

    def to_dict(self) -> Dict:
        data = {"model": self.model.value.model_name, "hws": self.hws.tolist(), "params": self.params.tolist()}
        return data


    @abstractmethod
    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def pixelwise_directions(self) -> torch.Tensor:
        uvs = self.pixelwise()  # batch_size + [h, w, 2]
        assert len(self.batch_size) == 1
        cam_indices = torch.arange(self.batch_size[0])
        cam_indices = cam_indices.view(-1, 1, 1).expand(uvs.shape[:-1]).long()
        cam = self[cam_indices]
        direction = cam.backproject_to_3d(uvs)
        return direction

    def is_same_size(self) -> bool:
        return torch.all(self.hws == self.hws[..., 0, :]).item()

    def pixelwise(self) -> torch.Tensor:
        if self.is_same_size():
            hw = self.hws[..., 0, :]
            h = hw[0].item()
            w = hw[1].item()
            xs = torch.linspace(0, w - 1, w, device=self.device)
            ys = torch.linspace(0, h - 1, h, device=self.device)
            uvs = torch.stack(torch.meshgrid([xs, ys], indexing="xy"), dim=-1)  # [h, w, 2]

            uvs = uvs.view(len(self.batch_size) * [1] + [h, w, 2]).expand(self.batch_size + (h, w, 2))
            return uvs
        else:
            raise NotImplementedError("cameras have different image shape.")

    def remap(self, new_camera: "Camera", rotation_curr2new: torch.Tensor, image: torch.Tensor):
        direction = new_camera.pixelwise_directions()
        rotation_new2curr = torch.inverse(rotation_curr2new)
        direction = (rotation_new2curr @ direction.unsqueeze(-1)).squeeze(-1)
        uv = self.project_to_2d(direction)
        image = torch.permute(image, (2, 0, 1)).unsqueeze(0) / 255.0
        new_image = remap_cubic(image, uv) * 255
        new_image = new_image.squeeze(0).permute(1, 2, 0).type(torch.uint8)
        return new_image

    def get_fov(self):
        uvs = []
        ids = []
        for i in range(self.hws.shape[0]):
            h, w = self.hws[i].tolist()
            uv = [[[1, h * 0.5], [w - 2, h * 0.5]], [[w * 0.5, 1], [w * 0.5, h - 2]]]
            uvs.append(uv)
            ids.extend([i] * 4)
        # [n, 2, 2, 2]
        uvs = torch.tensor(uvs).to(self.device).float()
        ids = torch.tensor(ids).to(self.device).view(-1, 2, 2)
        cam = self[ids]
        # [n, 2, 2, 3]
        rays = cam.backproject_to_3d(uvs)

        rays = F.normalize(rays, dim=-1)

        fov = torch.acos(torch.sum(rays[..., 0, :] * rays[..., 1, :], dim=-1))
        return fov
    
    def depth_to_pointcloud(self, depth_map: torch.Tensor):
        '''z-depth map to point cloud.
        depth map: [N, h, w, 1]
        '''
        if depth_map.ndim == 2:
            depth_map = depth_map.squeeze(0).unsqueeze(-1)

        directions = self.pixelwise_directions().squeeze(0)  # [n, h, w, 3]
        xyz = directions / directions[..., 2:] * depth_map
        return xyz

def remap_cubic(img: torch.Tensor, uv: torch.Tensor, border_mode: str = "border") -> torch.Tensor:
    """Remap image using bicubic interpolation.

    Args:
        img (torch.Tensor): Image tensor
        map_x (torch.Tensor): x mapping
        map_y (torch.Tensor): y mapping
        border_mode (str, optional): What to do with borders. Defaults to "border".

    Returns:
        torch.Tensor: _description_
    """
    batch_size, channels, height, width = img.shape

    grid = (uv / torch.tensor([width, height], device=img.device)) * 2 - 1

    if border_mode == "border":
        grid = torch.clamp(grid, -1, 1)
    elif border_mode == "wrap":
        grid = torch.remainder(grid + 1, 2) - 1

    # grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")


def create_camera(type: Union[CameraModel, str], hws: torch.Tensor, params: torch.Tensor) -> Camera:

    if isinstance(type, CameraModel):
        camera_name = type.value.model_name
    else:
        assert(isinstance(type, str))
        camera_name = type

    batch_size = hws.shape[:-1]
    return _camera_cls_[camera_name](
        hws=hws, params=params, batch_size=batch_size, model=_camera_cls_[camera_name].model
    )


def create_camera_from_dict(data: Dict) -> Camera:
    model_name = data["model"]
    model = _camera_cls_[model_name].model
    hws = torch.tensor(data["hws"]).view(-1, 2)
    if model.value.num_params != 0:
        params = torch.tensor(data["params"]).view(-1, model.value.num_params)
        assert hws.shape[:-1] == params.shape[:-1]
    else:
        params = torch.tensor(hws.shape[:-1] + (0,))

    return create_camera(model, hws, params)


if __name__ == "__main__":
    hw = torch.tensor([2, 3])
    h = hw[0].item()
    w = hw[1].item()

    xs = torch.linspace(0, w - 1, w)
    ys = torch.linspace(0, h - 1, h)
    uvs = torch.stack(torch.meshgrid([xs, ys], indexing="xy"), dim=-1)
    batch_size = [100, 5]
    uvs = uvs.view(len(batch_size) * [1] + [h, w, 2]).expand(batch_size + [h, w, 2])

    print(uvs.shape, uvs)
