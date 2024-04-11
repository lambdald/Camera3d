import torch
from enum import Enum
from tensordict import tensorclass


class CoordinateType(Enum):
    OpenCV = "OpenCV"
    OpenGL = "OpenGL"


@tensorclass
class Transform3d:
    transforms: torch.Tensor
    """shape: [batchsize] + [4, 4]"""

    coord_type: CoordinateType

    def inverse(self):
        return Transform3d(transforms=self.transforms.inverse(), coord_type=CoordinateType, batch_size=self.batch_size)
