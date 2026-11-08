# Efficient Sparse Attention for Long-Context Transformers

**This repository contains the official implementation of the paper
"Efficient Sparse Attention for Long-Context Transformers" (ACL 2026).**

The code covers data preparation, training of the routing network, and
evaluation on the long-context benchmarks reported in the paper.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
bash scripts/train_sparse_attention.sh --config configs/acl_sparse.yaml
```

## Evaluation

```bash
python -m sparse_attention.evaluate --checkpoint checkpoints/sparse_acl.pt
```

## Citation

```bibtex
@inproceedings{cook2026sparse,
  title={Efficient Sparse Attention for Long-Context Transformers},
  author={Cook, Ronald and Zhang, Alice and Yilmaz, Mehmet},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```
