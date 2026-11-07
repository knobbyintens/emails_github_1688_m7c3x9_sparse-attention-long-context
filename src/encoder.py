"""Vision-language encoder with sparse expert routing."""

import torch
import torch.nn as nn

from src.mixture_router import SparseMixtureRouter, load_balance_loss


class VisionLanguageEncoder(nn.Module):
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(hidden_dim, 8, batch_first=True),
                        "router": SparseMixtureRouter(hidden_dim, num_experts, top_k),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.num_experts = num_experts

    def forward(self, vision, text):
        hidden = torch.cat([self.vision_proj(vision), self.text_proj(text)], dim=1)
        aux_losses = []
        for layer in self.layers:
            attended, _ = layer["self_attn"](hidden, hidden, hidden)
            hidden = hidden + attended
            routed, indices = layer["router"](hidden)
            hidden = hidden + routed
            aux_losses.append(load_balance_loss(indices, self.num_experts))
        return hidden, torch.stack(aux_losses).mean()
