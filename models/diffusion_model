# models/diffusion_model.py
import torch.nn as nn
from .complex_layers import ComplexLinear, NaiveComplexLayerNorm, DiA, DiffusionEmbedding, ComplexSiLU, PositionEmbedding, DiffusionEmbedding

class tfdiff_WiFi(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.pos_embed = PositionEmbedding(params.sample_rate, params.input_dim, params.hidden_dim)
        self.t_embed = DiffusionEmbedding(params.max_step, params.embed_dim, params.hidden_dim)
        self.c_embed = nn.Sequential(ComplexLinear(params.feature_dim, params.hidden_dim), ComplexSiLU(), ComplexLinear(params.hidden_dim, params.hidden_dim))
        self.blocks = nn.ModuleList([DiA(params.hidden_dim, params.num_heads) for _ in range(params.num_block)])
        self.final_layer = ComplexLinear(params.hidden_dim, params.input_dim)
    def forward(self, x, t, c):
        x = self.pos_embed(x)
        t_emb = self.t_embed(t)
        c_emb = self.c_embed(c)
        c_combined = t_emb + c_emb
        for block in self.blocks:
            x = block(x, c_combined)
        return self.final_layer(x)
    