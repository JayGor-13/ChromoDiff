import math
import torch
import torch.nn as nn
from .embedding import SinusoidalPositionEmbeddings


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPESelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm = nn.GroupNorm(8, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = x.permute(0, 2, 1)

        qkv = self.qkv_proj(h)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        cos, sin = self.rope(q)
        cos = cos.to(q.device)
        sin = sin.to(q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, C)

        out = h + self.out_dropout(self.out_proj(attn_out))
        out = out.permute(0, 2, 1)
        return self.norm(out)

class DilatedResidualBlock(nn.Module):
    """
    1D dilated residual block with time conditioning.
    Uses GroupNorm (not BatchNorm) for stable training across varying
    diffusion corruption levels — BatchNorm statistics shift with t.
    """
    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        # Padding = dilation maintains input sequence length (1024) for kernel_size=3
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.act1 = nn.GELU()
        
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        
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
    def __init__(
        self,
        vocab_size: int = 6,
        hidden_dim: int = 256,
        dilations: list = None,
        attention_dropout: float = 0.1,
    ):
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
        self.attn = RoPESelfAttention(hidden_dim, num_heads=4, dropout=attention_dropout)

        self.output_norm = nn.GroupNorm(8, hidden_dim)
        self.final_conv = nn.Conv1d(hidden_dim, vocab_size - 1, kernel_size=1)

    def forward(self, noisy_dna: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Input DNA shapes: [B, L] -> [B, L, C] -> [B, C, L]
        x = self.dna_embedding(noisy_dna).permute(0, 2, 1)
        
        # Time Embedding projection
        t_emb = self.time_mlp(t)

        # Res blocks forwarding
        for block in self.res_blocks:
            x = block(x, t_emb)

        x = self.attn(x)

        # Final projection to vocabulary classes
        logits = self.final_conv(self.output_norm(x)) # [B, 5, L]
        return logits
