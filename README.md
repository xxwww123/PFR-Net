# PFR-Net Reference Modules

This repository provides the method-level reference modules for PFR-Net.

The release contains the principal components described in the paper:

- Intervention Discovered Expert Decomposition (IDED)
- Masked Cross-Expert Disentangling (MCED)
- shared part queries
- reliability-conditioned part consensus (RCCI)

## Scope

The repository is intended to document the main method design and interfaces.
It is not a complete training or reproduction package.

The following components are not included:

- dataset preparation and artifact-level split construction;
- backbone implementation and model assembly;
- intervention generation;
- spatial mask generation;
- classifier and auxiliary heads;
- counterfactual part-removal sampling;
- complete training objective and optimization loop;
- test-time augmentation and evaluation scripts;
- baseline implementations.

## Structure

```text
PFRNet_Public_Reference/
├── README.md
├── requirements.txt
├── pyproject.toml
└── pfrnet_modules/
    ├── __init__.py
    ├── config.py
    ├── ided.py
    ├── mced.py
    └── rcci.py
```

## Requirements

- Python 3.10+
- PyTorch 2.5+

```bash
pip install -r requirements.txt
```

## Import

```python
from pfrnet_modules import (
    ExpertAdapter,
    CrossPredictor,
    SharedPartQueries,
    ReliabilityHead,
)
```

The code is provided as a compact reference for the proposed modules.
