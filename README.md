# Sparse Mixture Routing for Efficient Vision-Language Pretraining

**Status: code released.**

This repository contains the implementation for the paper:

> **Sparse Mixture Routing for Efficient Vision-Language Pretraining**
> Maria Hernandez, Wei Zhang, Sofia Alvarez
> CVPR 2024

The released code contains the sparse mixture router, the vision-language
encoder and the pretraining/evaluation scripts used in the paper.

## Repository layout

- `src/` - model, data pipeline, training and evaluation code
- `configs/` - experiment configuration files
- `scripts/` - launch scripts

## Getting started

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
```

## Citation

If you use this code, please cite the paper above.
