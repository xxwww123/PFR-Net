from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CrossPredictor(nn.Module):
    """Cross-expert predictor used by MCED."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2

        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=hidden,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, source: Tensor, mask: Tensor) -> Tensor:
        source = source.detach()
        return self.net(torch.cat((source * mask, mask), dim=1))


def projected_subtraction(
    target: Tensor,
    prediction: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Remove the bounded projection of cross-predictable content."""

    prediction = prediction.detach()
    numerator = (target * prediction).sum(dim=(-2, -1), keepdim=True)
    denominator = prediction.square().sum(dim=(-2, -1), keepdim=True)
    alpha = (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0)
    return target - alpha * prediction, alpha


def orthogonality_loss(texture: Tensor, structure: Tensor) -> Tensor:
    texture = texture.flatten(1)
    structure = structure.flatten(1)
    similarity = F.cosine_similarity(texture, structure, dim=1)
    return similarity.square().mean()
