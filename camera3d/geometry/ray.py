import torch
from dataclasses import dataclass
from typing import Optional
from tensordict import tensorclass

@tensorclass
class Ray:
    origin: torch.Tensor  # [..., 3]
    direction: torch.Tensor  # [..., 3]
