#!/usr/bin/env bash
set -euo pipefail
python -m sparse_attention.train --config "$1"
