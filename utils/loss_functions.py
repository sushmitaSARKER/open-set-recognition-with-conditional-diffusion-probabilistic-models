import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)
        dot_product = torch.sum(torch.conj(a_flat) * b_flat, dim=1)
        norm_a = torch.linalg.norm(a_flat, dim=1)
        norm_b = torch.linalg.norm(b_flat, dim=1)
        cos_sim = dot_product.real / (norm_a * norm_b + 1e-8)
        return torch.mean(cos_sim)