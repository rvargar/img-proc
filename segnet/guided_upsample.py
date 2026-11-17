"""Image-guided upsampling of the logits: the difference filter used as the last layer.

The network predicts at 1/4 resolution, and bilinear upsampling places every class border
without looking at the image. Here every full-resolution pixel takes a weighted mean of the
(bilinearly upsampled) logits at the points of a circular sampling pattern, and a point
counts the more, the more similar its colour is to the centre pixel:

    score_i = -|g(p + o_i) - g(p)|_1 / sigma_circle(i) + bias_i        score_centre = centre_bias
    out(p)  = sum_i softmax(score)_i * logits(p + o_i)

In the terms of generalFilter this is gamma = |v - c| on the image, and alpha = a soft
pooling over the points whose weights come from the image and whose values are the logits.
Learned: the guide g (a colour mix of RGB), sigma per circle, a bias per point and the bias of
the centre. The offsets are fixed integers, so a point is a shifted view of the logits and
costs no memory. Related: joint bilateral upsampling, pixel-adaptive convolution.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedUpsample(nn.Module):
    """logits (B, K, h, w) + rgb (B, 3, H, W) in [0, 1]  ->  logits (B, K, H, W).

    radii:       radius of each circle of the pattern, in full-resolution pixels
    directions:  points per circle
    sigma:       initial colour distance (sum over the guide channels) at which the weight of
                 a point drops to 1/e
    centre_bias: initial score of the centre pixel; a point of equal colour has score 0
    """

    def __init__(self, radii=(2, 4), directions=8, guide_channels=3, sigma=0.1, centre_bias=2.0):
        super().__init__()
        offsets, circle = [], []
        for c, r in enumerate(radii):
            for j in range(directions):
                t = 2 * math.pi * j / directions
                o = (int(round(r * math.cos(t))), int(round(r * math.sin(t))))
                if o != (0, 0) and o not in offsets:
                    offsets.append(o)
                    circle.append(c)
        self.offsets, self.circle = offsets, circle
        self.pad = max(max(abs(dx), abs(dy)) for dx, dy in offsets)
        guide = torch.zeros(guide_channels, 3, 1, 1)
        guide[:, :, 0, 0] = torch.eye(3)[torch.arange(guide_channels) % 3]
        self.guide = nn.Parameter(guide)
        self.log_sigma = nn.Parameter(torch.full((len(radii),), math.log(sigma)))
        self.bias = nn.Parameter(torch.zeros(len(offsets)))
        self.centre_bias = nn.Parameter(torch.tensor(float(centre_bias)))

    def state(self):
        return {'guide': self.guide.detach()[:, :, 0, 0].cpu().tolist(),
                'sigma': self.log_sigma.detach().exp().cpu().tolist(),
                'bias': self.bias.detach().cpu().tolist(), 'centre_bias': self.centre_bias.item(),
                'offsets': self.offsets}

    def weights(self, rgb):
        """(B, 1 + n_points, H, W) weights of the centre and of every point; they sum to 1."""
        B, _, H, W = rgb.shape
        p = self.pad
        with torch.cuda.amp.autocast(enabled=False):
            g = F.conv2d(rgb.float(), self.guide)
            gp = F.pad(g, (p, p, p, p), mode='replicate')
            inv_sigma = torch.exp(-self.log_sigma)
            scores = [self.centre_bias.expand(B, 1, H, W)]
            for i, (dx, dy) in enumerate(self.offsets):
                d = (gp[:, :, p + dy:p + dy + H, p + dx:p + dx + W] - g).abs().sum(dim=1, keepdim=True)
                scores.append(self.bias[i] - d * inv_sigma[self.circle[i]])
            return torch.softmax(torch.cat(scores, dim=1), dim=1)

    def forward(self, logits, rgb):
        H, W = rgb.shape[2:]
        p = self.pad
        up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        w = self.weights(rgb).to(up.dtype)
        upp = F.pad(up, (p, p, p, p), mode='replicate')
        out = w[:, 0:1] * up
        for i, (dx, dy) in enumerate(self.offsets):
            out = out + w[:, i + 1:i + 2] * upp[:, :, p + dy:p + dy + H, p + dx:p + dx + W]
        return out
