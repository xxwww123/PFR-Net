from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ExpertAdapter(nn.Module):
    """Residual expert adapter used by IDED."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 4

        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        residual = self.adapter(feature)
        refined = feature + self.scale * residual
        return residual, refined


def role_margin_loss(
    d_texture_on_texture: Tensor,
    d_texture_on_structure: Tensor,
    d_structure_on_texture: Tensor,
    d_structure_on_structure: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Relative response ordering used for expert-role supervision."""

    texture_term = F.relu(
        margin - (d_texture_on_texture - d_texture_on_structure)
    )
    structure_term = F.relu(
        margin - (d_structure_on_structure - d_structure_on_texture)
    )
    return (texture_term + structure_term).mean()
