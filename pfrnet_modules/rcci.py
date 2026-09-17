from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SharedPartQueries(nn.Module):
    """Shared query bank for aligned local evidence."""

    def __init__(self, channels: int, num_parts: int = 6) -> None:
        super().__init__()
        self.num_parts = num_parts
        self.queries = nn.Parameter(torch.randn(num_parts, channels) * 0.02)

    def forward(self, residual: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, channels, height, width = residual.shape

        logits = torch.einsum(
            "pc,bchw->bphw",
            self.queries,
            residual,
        ) / math.sqrt(channels)

        attention = F.softmax(
            logits.flatten(-2),
            dim=-1,
        ).view(batch, self.num_parts, height, width)

        part_residuals = (
            height * width * attention[:, :, None] * residual[:, None]
        )
        descriptors = part_residuals.mean(dim=(-2, -1))
        return attention, part_residuals, descriptors


class ReliabilityHead(nn.Module):
    """Part-level utility and utility-error estimator."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 4

        self.shared = nn.Sequential(
            nn.Linear(3 * channels, hidden),
            nn.ReLU(inplace=True),
        )
        self.utility = nn.Linear(hidden, 1)
        self.error = nn.Linear(hidden, 1)

    def forward(
        self,
        current: Tensor,
        paired: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        descriptor = torch.cat(
            (current, paired, (current - paired).abs()),
            dim=-1,
        )
        feature = self.shared(descriptor)

        utility = self.utility(feature).squeeze(-1)
        utility_error = F.softplus(self.error(feature).squeeze(-1))
        return utility, utility_error


def reliability_consensus(
    texture_parts: Tensor,
    structure_parts: Tensor,
    utility: Tensor,
    utility_error: Tensor,
    beta: float = 0.5,
    temperature: float = 0.1,
    tau: float = 0.5,
    utility_boundary: float = 0.0,
) -> Dict[str, Tensor]:
    """Reliability-conditioned part consensus."""

    score = utility - beta * utility_error

    availability = torch.sigmoid(
        (score - utility_boundary) / temperature
    )

    log_weight = (
        torch.log(availability.clamp_min(1e-8))
        + score / tau
    )
    expert_weight = F.softmax(log_weight, dim=1)

    part_gate = 1.0 - torch.prod(
        1.0 - availability,
        dim=1,
    )

    parts = torch.stack(
        (texture_parts, structure_parts),
        dim=1,
    )

    weighted_parts = (
        expert_weight[:, :, :, None, None, None] * parts
    ).sum(dim=1)

    consensus = (
        part_gate[:, :, None, None, None] * weighted_parts
    ).mean(dim=1)

    return {
        "consensus": consensus,
        "score": score,
        "availability": availability,
        "expert_weight": expert_weight,
        "part_gate": part_gate,
    }
