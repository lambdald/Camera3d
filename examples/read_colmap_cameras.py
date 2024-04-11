from camera3d import read_colmap_cameras
from pathlib import Path

colmap_camera_filepath = Path('/home/lidong/data/360_v2/bicycle/sparse/0/cameras.bin')

cameras = read_colmap_cameras(colmap_camera_filepath)
print(cameras)