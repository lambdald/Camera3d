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

## Installation

Make sure you have installed cuda and pytorch.

```bash
git clone https://github.com/lambdald/Camera3d
cd Camera3d
pip install .
```

## Examples

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

More examples can be found in the examples folder.
