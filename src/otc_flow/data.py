"""Manifest datasets for the public OTC-Flow release.

Each JSONL row contains paths relative to the manifest directory:
``ct``, ``tee`` and ``conditions``.  ``labels`` and ``ct_mask`` are optional
training targets. Tensor files are stored as ``.pt`` or ``.npy``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".npy":
        value = torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported tensor file: {path}")
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


class ManifestDataset(Dataset[dict[str, torch.Tensor]]):
    """Load fixed-shape CT/TEE training examples from a JSONL manifest."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row: dict[str, Any] = self.rows[index]
        sample = {
            "ct_pixel_values": _load_tensor(self.root / row["ct"]).float(),
            "real_tee": _load_tensor(self.root / row["tee"]).float(),
            "conditions": _load_tensor(self.root / row["conditions"]).float(),
        }
        for name in ("labels", "ct_mask"):
            if name in row:
                sample[name] = _load_tensor(self.root / row[name]).float()
        return sample


class ImageDataset(Dataset[torch.Tensor]):
    """Load VQ-VAE images from a JSONL manifest containing an ``image`` path."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        value = _load_tensor(self.root / self.rows[index]["image"]).float()
        return value if value.ndim == 3 else value.unsqueeze(0)
