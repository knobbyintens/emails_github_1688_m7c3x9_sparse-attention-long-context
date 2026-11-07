"""Top-k sparse mixture routing for vision-language experts."""

import torch
import torch.nn as nn


class SparseMixtureRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(tokens)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        output = torch.zeros_like(tokens)
        for slot in range(self.top_k):
            expert_ids = indices[..., slot]
            slot_weights = weights[..., slot].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if not mask.any():
                    continue
                expert_input = tokens[mask]
                output[mask] += slot_weights[mask] * self.experts[expert_id](expert_input)
        return output, indices


def load_balance_loss(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.bincount(indices.flatten(), minlength=num_experts).float()
    fractions = counts / counts.sum()
    target = torch.full_like(fractions, 1.0 / num_experts)
    return torch.sum((fractions - target) ** 2)
