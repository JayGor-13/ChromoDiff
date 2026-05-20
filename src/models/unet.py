import torch
import torch.nn as nn
from .embedding import SinusoidalPositionEmbeddings

class DilatedResidualBlock(nn.Module):
    """
    1D dilated residual block with time conditioning.
    Receptive field = 2 * dilation + 1. Replicates spatial dimension of 1024.
    """
    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        # Padding = dilation maintains input sequence length (1024) for kernel_size=3
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # 1. Dilated Conv 1
        h = self.conv1(x)
        # 2. Normalization & Activation
        h = self.act1(self.norm1(h))
        # 3. Time Projection (added element-wise)
        t_proj = self.time_proj(t_emb).unsqueeze(2) # [B, C, 1]
        h = h + t_proj
        # 4. Dilated Conv 2 & Normalization & Activation
        h = self.act1(self.norm2(self.conv2(h)))
        # 5. Residual Connection
        return x + h

class GenoDiff1D(nn.Module):
    """
    Symmetric 1D Dilated Residual CNN model for DNA sequence denoising.
    """
    def __init__(self, vocab_size: int = 6, hidden_dim: int = 256, dilations: list = None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32] # 6 residual blocks as specified in Section 4

        self.dna_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.res_blocks = nn.ModuleList([
            DilatedResidualBlock(hidden_dim, dilation=d) for d in dilations
        ])

        self.output_norm = nn.BatchNorm1d(hidden_dim)
        self.final_conv = nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)

    def forward(self, noisy_dna: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Input DNA shapes: [B, L] -> [B, L, C] -> [B, C, L]
        x = self.dna_embedding(noisy_dna).permute(0, 2, 1)
        
        # Time Embedding projection
        t_emb = self.time_mlp(t)

        # Res blocks forwarding
        for block in self.res_blocks:
            x = block(x, t_emb)

        # Final projection to vocabulary classes
        logits = self.final_conv(self.output_norm(x)) # [B, 6, L]
        return logits
