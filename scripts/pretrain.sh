#!/usr/bin/env bash
set -euo pipefail
python -m src.pretrain --config configs/pretrain.yaml --steps 10000
