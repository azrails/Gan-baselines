import torch
import torch.nn as nn
from einops import rearrange

class Attn(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=.1):
        super().__init__()
        is_project = not (heads == 1 and dim == head_dim)
        self.heads = heads
        self.inner_dim = heads * head_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, self.inner_dim * 3)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(
            self.inner_dim, dim
            ) if is_project else nn.Identity()
    
    def forward(self, x):
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b s (n d) -> b n s d', n=self.heads, d=self.head_dim), qkv)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        probs = self.dropout(torch.softmax(scores, dim=-1))
        logits = torch.matmul(probs, v)
        logits = rearrange(logits, 'b n s d -> b s (n d)', n=self.heads, d=self.head_dim)
        logits = self.out_proj(logits)
        return logits


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.ffn(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_ratio, dropout=.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
                Attn(dim, heads, head_dim, dropout),
                FFN(dim, int(dim * mlp_ratio))
        ])

    def forward(self, x):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c', c=c, b=b, h=h, w=w)
        attn, ffn = self.layers
        x = attn(x) + x
        x = ffn(x) + x
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', b=b, c=c, h=h, w=w)
        return x
        


