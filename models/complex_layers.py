# models/complex_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# All the complex layer classes here: ComplexLinear, ComplexConv1d, 
#  ComplexSiLU, NaiveComplexLayerNorm, ComplexMultiHeadAttention, etc.

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        return torch.complex(self.fc_r(x.real) - self.fc_i(x.imag), self.fc_r(x.imag) + self.fc_i(x.real))

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        return torch.complex(self.conv_r(x.real) - self.conv_i(x.imag), self.conv_r(x.imag) + self.conv_i(x.real))

class ComplexSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x.real) + 1j * F.silu(x.imag)
        
class NaiveComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.ln_r = nn.LayerNorm(normalized_shape, eps=eps)
        self.ln_i = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, x):
        return torch.complex(self.ln_r(x.real), self.ln_i(x.imag))
    
# --- RF-Diffusion Model Definitions ---
@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(ComplexLinear(embed_dim, hidden_dim), ComplexSiLU(), ComplexLinear(hidden_dim, hidden_dim))
    def forward(self, t):
        x = self.embedding[t]
        return self.projection(x)
    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)
        dims = torch.arange(embed_dim).unsqueeze(0)
        table = steps * torch.exp(-math.log(10000) * dims / (embed_dim))
        return torch.exp(1j*table)

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.projection = ComplexLinear(input_dim, hidden_dim)
    def forward(self, x):
        return self.projection(x)    

class ComplexMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, q, k, v):
        attn_output_real, _ = self.mha(q.real, k.real, v.real)
        attn_output_imag, _ = self.mha(q.imag, k.imag, v.imag)
        return torch.complex(attn_output_real, attn_output_imag)

class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = NaiveComplexLayerNorm(hidden_dim)
        self.attn = ComplexMultiHeadAttention(hidden_dim, num_heads)
        self.adaLN_modulation = nn.Sequential(ComplexSiLU(), ComplexLinear(hidden_dim, 2 * hidden_dim))
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        mod_x = modulate(self.norm1(x), shift, scale)
        attn_output = self.attn(mod_x, mod_x, mod_x)
        return x + attn_output


def complex_leaky_relu(z, negative_slope=0.3):
    return F.leaky_relu(z.real, negative_slope) + 1j * F.leaky_relu(z.imag, negative_slope)
