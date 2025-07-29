

import torch.nn as nn
from .complex_layers import ComplexLinear, ComplexConv1d, NaiveComplexLayerNorm, complex_leaky_relu
    

class DisentangledFeatureExtractor(nn.Module):
    def __init__(self, num_classes=3, feature_dim=16):
        super().__init__()
        self.conv_t1 = ComplexConv1d(1, 8, kernel_size=3, stride=2, padding=1)
        self.norm_t1 = NaiveComplexLayerNorm([8, 256])
        self.conv_t2 = ComplexConv1d(8, feature_dim, kernel_size=3, stride=2, padding=1)
        self.norm_t2 = NaiveComplexLayerNorm([feature_dim, 128])
        
        self.conv_f1 = ComplexConv1d(1, 8, kernel_size=3, stride=2, padding=1)
        self.norm_f1 = NaiveComplexLayerNorm([8, 256])
        self.conv_f2 = ComplexConv1d(8, feature_dim, kernel_size=3, stride=2, padding=1)
        self.norm_f2 = NaiveComplexLayerNorm([feature_dim, 128])
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_time = ComplexLinear(feature_dim, num_classes)
        self.classifier_freq = ComplexLinear(feature_dim, num_classes)
        self.classifier_combined = ComplexLinear(feature_dim, num_classes)

    def forward(self, x_time, x_freq):
        x_time = x_time.unsqueeze(1)
        x_freq = x_freq.unsqueeze(1)
        z_t = complex_leaky_relu(self.norm_t1(self.conv_t1(x_time)))
        z_t = complex_leaky_relu(self.norm_t2(self.conv_t2(z_t)))
        z_f = complex_leaky_relu(self.norm_f1(self.conv_f1(x_freq)))
        z_f = complex_leaky_relu(self.norm_f2(self.conv_f2(z_f)))
        out_t = self.avg_pool(z_t).squeeze(-1)
        out_f = self.avg_pool(z_f).squeeze(-1)
        features_for_diffusion = 0.5 * out_t + 0.5 * out_f
        logits_time = self.classifier_time(out_t)
        logits_freq = self.classifier_freq(out_f)
        logits_combined = self.classifier_combined(features_for_diffusion)
        return [logits_time.abs(), logits_freq.abs(), logits_combined.abs()], [z_t, z_f], features_for_diffusion