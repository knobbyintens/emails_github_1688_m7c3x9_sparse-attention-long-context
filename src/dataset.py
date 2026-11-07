"""Dataset utilities for multimodal dialogue."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DialogueGroundingDataset(Dataset):
    def __init__(self, manifest_path: str, max_turns: int = 24) -> None:
        self.root = Path(manifest_path).parent
        with open(manifest_path, encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.records)

    def _pad(self, tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.size(0) >= length:
            return tensor[:length]
        pad = torch.zeros(length - tensor.size(0), tensor.size(1))
        return torch.cat([tensor, pad], dim=0)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        text = torch.tensor(record["text_features"], dtype=torch.float32)
        vision = torch.tensor(record["vision_features"], dtype=torch.float32)
        return {
            "text_states": self._pad(text, self.max_turns),
            "vision_states": vision,
            "labels": torch.tensor(record["label"], dtype=torch.long),
        }
