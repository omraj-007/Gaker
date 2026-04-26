from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from defense.utils import read_jsonl


@dataclass
class DefenseSample:
    adv: torch.Tensor
    source: torch.Tensor
    target: Optional[torch.Tensor]
    source_label: int
    target_label: int
    sample_id: int
    adv_path: str
    source_path: str
    target_path: Optional[str]


class DefenseDataset(Dataset):
    def __init__(
        self,
        split_root: str,
        image_size: Optional[int] = None,
        return_target: bool = False,
    ):
        self.split_root = Path(split_root)
        self.records: List[Dict] = read_jsonl(self.split_root / "metadata.jsonl")
        self.return_target = bool(return_target)
        self.transform = self._build_transform(image_size=image_size)

    def _build_transform(self, image_size: Optional[int]):
        if image_size is None or int(image_size) <= 0:
            return transforms.Compose([transforms.ToTensor()])
        return transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.split_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing image referenced in split metadata: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx: int) -> DefenseSample:
        record = self.records[idx]
        target_rel = record.get("target_image")
        target_tensor = self._load_image(target_rel) if self.return_target and target_rel is not None else None
        return DefenseSample(
            adv=self._load_image(record["adv_image"]),
            source=self._load_image(record["source_image"]),
            target=target_tensor,
            source_label=int(record["source_label"]),
            target_label=int(record["target_label"]),
            sample_id=int(record.get("sample_id", idx)),
            adv_path=str(self.split_root / record["adv_image"]),
            source_path=str(self.split_root / record["source_image"]),
            target_path=str(self.split_root / target_rel) if target_rel is not None else None,
        )


def defense_collate(batch: List[DefenseSample]) -> Dict[str, torch.Tensor | List[str] | List[int] | None]:
    adv = torch.stack([item.adv for item in batch], dim=0)
    source = torch.stack([item.source for item in batch], dim=0)
    target = None
    if batch[0].target is not None:
        target = torch.stack([item.target for item in batch], dim=0)

    return {
        "adv": adv,
        "source": source,
        "target": target,
        "source_label": torch.tensor([item.source_label for item in batch], dtype=torch.long),
        "target_label": torch.tensor([item.target_label for item in batch], dtype=torch.long),
        "sample_id": [item.sample_id for item in batch],
        "adv_path": [item.adv_path for item in batch],
        "source_path": [item.source_path for item in batch],
        "target_path": [item.target_path for item in batch],
    }
