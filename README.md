# Camera3d
A 3D camera library for 3d vision based on pytorch and cuda.

This library is very useful in 3D vision, such as camera calibration, point cloud fusion, nerf, 3D gaussian splatting, etc.

Supported camera models:
* Pinhole
* FOV
* Simple Radial
* OpenCV
* OpenCV Fisheye
* Panoramic[Equirectangular Projection]

Supported features:
* Project 3d points to pixel coordinates of a specific camera model in real time.
* Calculate the 3D rays corresponding to the pixel coordinates of a specific camera model in real time.
* Undistort image by remapping pixels between different camera models.

Camera3d also supports reading colmap camera files. Currently, it supports four camera models: pinhole, simple radial, opencv, and opencv fisheye.


The camera class in Camera3d is a [TensorDict](https://github.com/pytorch/tensordict?tab=readme-ov-file#tensordict) object, so it possesses all the basic features of torch.Tensor such as clear, copy, fromkeys, get, items, keys, pop, popitem, setdefault, update and values.


## Installation

Make sure you have installed cuda and pytorch.

```bash
git clone https://github.com/lambdald/Camera3d
cd Camera3d
pip install .
```

## Examples

Construct a camera group and randomly generate pixel coordinates and their corresponding camera indexes. Get the camera according to the camera index, and then calculate the ray corresponding to each pixel.

```python
from camera3d.camera import Camera, create_camera_from_dict, CameraModel
import torch

# two pinhole camera
cam_json = {
    "hws": [[100, 200], [200, 500]],
    "params": [[200.0, 200.0, 50, 100.0], [300, 300, 100, 250]],
    "model": "pinhole",
}
cam = create_camera_from_dict(cam_json)


N = 1920 * 1080
indices = torch.randint(0, 2, (N,), device="cpu")
# TensorDict indexing
cams = cam[indices].cuda()


whs = torch.flip(cams.hws, [-1])

# back-project pixels to rays.
uvs = torch.rand((N, 2), device="cuda") * (whs - 1)
rays = cams.backproject_to_3d(uvs)

# project 3d points to pixels.
uv_proj = cams.project_to_2d(rays)

# project error
diff = (uv_proj - uvs).norm(dim=-1)
print("max project error=", diff.max().item())
print("eps=", torch.finfo(torch.float32).eps)

```

Read colmap camera files
```python

from camera3d import read_colmap_cameras
from pathlib import Path

colmap_camera_filepath = Path('sparse/0/cameras.bin')

cameras = read_colmap_cameras(colmap_camera_filepath)
print(cameras)
```


More examples can be found in the examples folder.
