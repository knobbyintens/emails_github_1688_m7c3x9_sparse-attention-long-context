"""Pretraining loop for the sparse vision-language model."""

import argparse

import torch
from torch.utils.data import DataLoader

from src.encoder import VisionLanguageEncoder
from src.dataset import PairedImageTextDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    dataset = PairedImageTextDataset("data/processed/pairs.jsonl")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionLanguageEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    step = 0
    model.train()
    while step < args.steps:
        for batch in loader:
            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            hidden, aux_loss = model(vision, text)
            contrastive = torch.nn.functional.cross_entropy(hidden @ hidden.T, torch.arange(hidden.size(0), device=device))
            loss = contrastive + 0.01 * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"step={step} loss={loss.item():.4f} aux={aux_loss.item():.4f}")
            step += 1
            if step >= args.steps:
                break
    torch.save(model.state_dict(), "outputs/sparse_mixture.pt")


if __name__ == "__main__":
    main()
