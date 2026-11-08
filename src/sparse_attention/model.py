"""Block-sparse attention with a learned routing network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingNetwork(nn.Module):
    """Predicts which KV blocks each query block should attend to."""

    def __init__(self, dim, num_blocks, hidden=128):
        super().__init__()
        self.num_blocks = num_blocks
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_blocks),
        )

    def forward(self, query_block):
        logits = self.scorer(query_block.mean(dim=1))
        return torch.sigmoid(logits)


class SparseBlockAttention(nn.Module):
    """Block-sparse multi-head attention used in the ACL 2026 paper."""

    def __init__(self, dim, num_heads=8, block_size=64, sparsity=0.5):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.sparsity = sparsity
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        num_blocks = math.ceil(seq_len / self.block_size)
        scale = 1.0 / math.sqrt(self.head_dim)

        outputs = []
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, seq_len)
            q_block = q[:, :, start:end]
            scores = torch.matmul(q_block, k.transpose(-1, -2)) * scale
            mask = self._block_mask(scores, num_blocks)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            outputs.append(torch.matmul(weights, v))
        attended = torch.cat(outputs, dim=2)
        attended = attended.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        return self.out(attended)

    def _block_mask(self, scores, num_blocks):
        # Keep the local block and the top-k routed blocks for every query block.
        keep = max(1, int(num_blocks * (1.0 - self.sparsity)))
        block_scores = scores.mean(dim=(0, 1))
        topk = torch.topk(block_scores, keep, dim=-1).indices
        mask = torch.ones_like(scores, dtype=torch.bool)
        for idx in topk:
            mask[:, :, idx] = False
        return mask
