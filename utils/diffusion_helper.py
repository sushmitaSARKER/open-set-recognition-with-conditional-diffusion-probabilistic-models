import torch
import math


class SignalDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.max_step = params.max_step
        beta = torch.tensor(params.noise_schedule)
        alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)
    def degrade_fn(self, x_0, t):
        device = x_0.device
        t_device = t.to(device)
        # Move the alpha_bar tensor to the same device as the indices before slicing
        alpha_bar_on_device = self.alpha_bar.to(device)
        info_weight = torch.sqrt(alpha_bar_on_device[t_device])
        noise_weight = torch.sqrt(1.0 - alpha_bar_on_device[t_device])
        noise = torch.randn_like(x_0)
        return x_0 * info_weight.view(-1, 1, 1) + noise * noise_weight.view(-1, 1, 1)
