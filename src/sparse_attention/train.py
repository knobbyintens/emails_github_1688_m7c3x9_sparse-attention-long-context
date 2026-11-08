"""Training entry point for the sparse attention experiments."""

import argparse

import torch
from torch.utils.data import DataLoader

from sparse_attention.model import SparseBlockAttention
from sparse_attention.data import LongContextDataset
from sparse_attention.routing import routing_loss


def train(config):
    dataset = LongContextDataset(config["data_path"], block_size=config["block_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    model = SparseBlockAttention(
        dim=config["dim"],
        num_heads=config["num_heads"],
        block_size=config["block_size"],
        sparsity=config["sparsity"],
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        for step, batch in enumerate(loader):
            tokens, labels = batch
            logits = model(tokens.cuda())
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.cuda().reshape(-1)
            )
            loss = loss + routing_loss(model, entropy_weight=config["entropy_weight"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
    torch.save(model.state_dict(), config["output_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        train(yaml.safe_load(handle))
