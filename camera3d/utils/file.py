"""
Description: filesystem
"""

from typing import List, Union
from pathlib import Path
import shutil
import os


def get_all_files(root: Union[str, Path], pattern: str) -> List[Path]:
    """get files in root dir, including files in subdirs."""
    root = Path(root)
    files = []
    for img_path in root.glob(pattern):
        files.append(img_path)

    for subdir in root.iterdir():
        if subdir.is_dir():
            files += get_all_files(subdir, pattern)
    return sorted(files)


def glob_imgs(path) -> List[Path]:
    imgs = []
    for ext in ["*.png", "*.PNG", "*.jpeg", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(get_all_files(path, ext))
    return imgs


def create_dir_if_not_exists(dir_path):
    lpath = Path(dir_path)
    if not lpath.exists():
        print("create dir:", dir_path)
        lpath.mkdir(parents=True)


def delete_if_exists(inpath):
    lpath = Path(inpath)
    if not lpath.exists():
        return

    if lpath.is_dir():
        shutil.rmtree(lpath)
    else:
        lpath.unlink()


def is_empty_dir(path):
    return len(os.listdir(path)) == 0
