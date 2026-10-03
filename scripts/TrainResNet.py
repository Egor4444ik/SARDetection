from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet34


@dataclass
class TrainConfig:
    data_root: Path
    out_dir: Path
    arch: str = "resnet18"
    num_classes: int = 40
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cpu"
    val_split: float = 0.1
    seed: int = 42


class SarPreprocessor:
    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        self._low = low
        self._high = high

    def to_rgb(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            arr = np.array(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, [self._low, self._high])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        uint8 = (arr * 255).astype(np.uint8)
        return Image.fromarray(np.stack([uint8] * 3, -1), mode="RGB")


class SarChipDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, int]],
        preprocessor: SarPreprocessor,
        transform: transforms.Compose,
    ) -> None:
        self._items = items
        self._pre = preprocessor
        self._tf = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        path, label = self._items[idx]
        rgb = self._pre.to_rgb(path)
        return self._tf(rgb), label


class DatasetSplitter:
    def __init__(self, root: Path, val_split: float, seed: int) -> None:
        self._root = root
        self._val_split = val_split
        self._seed = seed

    def split(self) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
        classes = sorted([d.name for d in self._root.iterdir() if d.is_dir()])
        cls_to_idx = {c: i for i, c in enumerate(classes)}

        train: list[tuple[Path, int]] = []
        val: list[tuple[Path, int]] = []
        rng = np.random.default_rng(self._seed)

        for cls in classes:
            files = sorted((self._root / cls).glob("*.tif"))
            idxs = rng.permutation(len(files))
            n_val = max(1, int(len(files) * self._val_split))
            val_idx = set(idxs[:n_val].tolist())
            for i, f in enumerate(files):
                (val if i in val_idx else train).append((f, cls_to_idx[cls]))

        return train, val, classes


class TransformBuilder:
    def __init__(self, size: int) -> None:
        self._size = size

    def train(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self._size, self._size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def eval(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self._size, self._size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class ModelFactory:
    @staticmethod
    def build(arch: str, num_classes: int) -> nn.Module:
        if arch == "resnet18":
            m = resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        if arch == "resnet34":
            m = resnet34(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        raise ValueError(f"unknown arch: {arch}")


class EpochRunner:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        self._model = model
        self._device = device
        self._criterion = criterion
        self._optimizer = optimizer

    def train(self, loader: DataLoader) -> tuple[float, float]:
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)
            self._optimizer.zero_grad()
            logits = self._model(imgs)
            loss = self._criterion(logits, labels)
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def eval(self, loader: DataLoader) -> tuple[float, float]:
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)
            logits = self._model(imgs)
            loss = self._criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)


class CheckpointSaver:
    def __init__(self, out_dir: Path, arch: str) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._arch = arch
        self._best = 0.0

    def maybe_save(self, model: nn.Module, acc: float, epoch: int) -> None:
        if acc <= self._best:
            return
        self._best = acc
        path = self._out_dir / f"{self._arch}_best.pth"
        torch.save(model.state_dict(), path)
        print(f"  saved best: epoch={epoch} acc={acc:.4f} -> {path}")


class TrainingPipeline:
    def __init__(self, config: TrainConfig) -> None:
        self._c = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self._device = torch.device(config.device)
        self._pre = SarPreprocessor()
        self._tf = TransformBuilder(config.image_size)

    def run(self) -> None:
        train_items, val_items, classes = DatasetSplitter(
            self._c.data_root, self._c.val_split, self._c.seed
        ).split()

        print(f"arch: {self._c.arch}")
        print(f"classes: {len(classes)}")
        print(f"train: {len(train_items)}, val: {len(val_items)}")

        train_ds = SarChipDataset(train_items, self._pre, self._tf.train())
        val_ds = SarChipDataset(val_items, self._pre, self._tf.eval())

        train_loader = DataLoader(
            train_ds, batch_size=self._c.batch_size,
            shuffle=True, num_workers=self._c.num_workers,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self._c.batch_size,
            shuffle=False, num_workers=self._c.num_workers,
        )

        model = ModelFactory.build(self._c.arch, self._c.num_classes)
        model.to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=self._c.lr,
            weight_decay=self._c.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self._c.epochs)

        runner = EpochRunner(model, self._device, criterion, optimizer)
        saver = CheckpointSaver(self._c.out_dir, self._c.arch)

        for epoch in range(1, self._c.epochs + 1):
            train_loss, train_acc = runner.train(train_loader)
            val_loss, val_acc = runner.eval(val_loader)
            scheduler.step()
            print(
                f"epoch {epoch:3d} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f}"
            )
            saver.maybe_save(model, val_acc, epoch)


def main() -> None:
    root = Path("data/_downloads/Ground_Range/Amplitude_8bit/extracted/SOC_40classes")
    for arch in ("resnet18", "resnet34"):
        cfg = TrainConfig(
            data_root=root / "train",
            out_dir=Path("runs/trained"),
            arch=arch,
            num_classes=40,
            image_size=224,
            batch_size=64,
            epochs=30,
            lr=1e-3,
            weight_decay=1e-4,
            num_workers=0,
            device="cpu",
        )
        TrainingPipeline(cfg).run()


if __name__ == "__main__":
    main()