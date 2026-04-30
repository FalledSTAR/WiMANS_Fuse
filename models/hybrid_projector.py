import torch
from torch import nn
import torch.nn.functional as F


class HybridProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 256, num_heads: int = 2):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected feature shape [B,D] or [B,N,D], got {tuple(x.shape)}")

        x = self.fc_in(x)
        x = F.relu(self.norm1(x))
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm2(x + attn_out)
        x = x.mean(dim=1)
        return self.fc_out(x)
