import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes discrete scalar timestep t -> [Batch, dim] sinusoidal dense vector.
    Allows the model blocks to adapt reconstruction depending on corruption rate.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        angles = time[:, None].float() * freq[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)
