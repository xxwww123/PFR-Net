from dataclasses import dataclass


@dataclass(frozen=True)
class MethodConfig:
    num_parts: int = 6
    role_margin: float = 0.10
    mask_ratio: float = 0.40

    beta: float = 0.50
    temperature: float = 0.10
    tau: float = 0.50
    utility_boundary: float = 0.0
