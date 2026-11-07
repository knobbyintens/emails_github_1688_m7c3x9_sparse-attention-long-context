"""Zero-shot retrieval evaluation."""

import torch


@torch.no_grad()
def evaluate_retrieval(model, loader, device):
    model.eval()
    recalls = {1: 0, 5: 0, 10: 0}
    total = 0
    for batch in loader:
        vision = batch["vision"].to(device)
        text = batch["text"].to(device)
        hidden, _ = model(vision, text)
        similarities = hidden @ hidden.T
        ranks = similarities.argsort(dim=-1, descending=True)
        for k in recalls:
            recalls[k] += (ranks[:, :k] == torch.arange(hidden.size(0), device=device).unsqueeze(1)).any(dim=1).sum().item()
        total += hidden.size(0)
    return {f"recall@{k}": v / max(total, 1) for k, v in recalls.items()}
