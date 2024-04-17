from pathlib import Path
import os
import sys
import torch
import rich
from rich.traceback import install

install(show_locals=False)

import torch

from camera3d.camera import Camera, create_camera_from_dict, CameraModel
from camera3d import FoVCamera

camera_dict = {
    "simple_radial": {
        "hws": [100, 200, 200, 500],
        "params": [200.0, 50, 100, 0.001, 300, 100, 250, 0.002],
        "model": "simple_radial",
    },
    "pinhole": {
        "hws": [100, 200, 200, 500],
        "params": [200.0, 200.0, 50, 100.0, 300, 300, 100, 250],
        "model": "pinhole",
    },
    "opencv": {
        "hws": [100, 200, 200, 500],
        "params": [
            200.0,
            200.0,
            50,
            100.0,
            0.1,
            0.15,
            0.13,
            0.11,
            0.1,
            300,
            300,
            100,
            250,
            0.1,
            0.2,
            0.15,
            0.01,
            0.1,
        ],
        "model": "opencv",
    },
    "opencv_fisheye": {
        "hws": [100, 200, 200, 500],
        "params": [200.0, 200.0, 50, 100.0, 0.1, 0.15, 0.13, 0.11, 300, 300, 100, 250, 0.1, 0.2, 0.15, 0.01],
        "model": "opencv_fisheye",
    },
    "panoramic": {"hws": [100, 200, 250, 500], "params": [], "model": "panoramic", "model_id": 5},
}


from rich import print
import pypose as pp

rand_pose = pp.randn_se3((2,)).matrix()


for cam_name, cam_json in camera_dict.items():
    print(cam_name)
    cam = create_camera_from_dict(cam_json)
    print(type(cam))
    print(cam.model)

    N = 1920 * 1080
    indices = torch.randint(0, 2, (N,), device="cpu")
    cams = cam[indices].cuda()

    print(cams.shape)
    whs = torch.flip(cams.hws, [-1])
    uvs = torch.rand((N, 2), device="cuda").clip(0.0, 1.0) * whs
    print(whs)
    print(uvs)

    import time

    t0 = time.time()
    rays = cams.backproject_to_3d(uvs)
    t1 = time.time()

    test_n = 10

    print("backproj:", t1 - t0, "s")
    t0 = time.time()
    for _ in range(test_n):
        uv_proj = cams.project_to_2d(rays)
    t1 = time.time()
    print("proj:", (t1 - t0) / test_n, "s")
    print("proj fps:", 1 / ((t1 - t0) / test_n))

    diff = (uv_proj - uvs).norm(dim=-1)
    if cam.model != CameraModel.Panoramic and cam.model != CameraModel.OpenCVFisheye:
        assert diff.max() < 1e-2, diff.max()

    print("diff max", diff.max())
    print("diff min", diff.min())
    # posed_camera = c_backend.PosedCamera(cam, rand_pose, c_backend.OpenCV)
    # ray_o, ray_d = posed_camera.GetRays(indices, uvs)
    # print(ray_o)
    # print(ray_d)

    # if cam_name == "pinhole":
    #     fov_cam = FoVCamera.from_camera(cam)
    #     print(fov_cam.get_projection_matrix(torch.randint(0, 2, (2,)), torch.rand((2, 2), device="cuda")))

    if cam.model == CameraModel.Pinhole:
        print("projection matrix:", cam.projection_matrix(0.1, 100))

    directions = cam[[0, 0, 0, 0]].cuda().pixelwise_directions()
    print(directions.shape)


    cam = cam[[0]].cuda()
    print(cam.shape)
    directions = cam.pixelwise_directions()
    uv = cam.project_to_2d(directions)
    ray = cam.backproject_to_3d(uv)

    diff = directions - ray
    print('max=', diff.max().item())