from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16

from scripts.TrainResNet import (
    CheckpointSaver,
    DatasetSplitter,
    EpochRunner,
    SarChipDataset,
    SarPreprocessor,
    TransformBuilder,
)


@dataclass
class ViTTrainConfig:
    data_root: Path
    out_dir: Path
    num_classes: int = 40
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 0
    device: str = "cpu"
    val_split: float = 0.1
    seed: int = 42


class ViTFactory:
    @staticmethod
    def build(num_classes: int) -> nn.Module:
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model


class ViTTrainingPipeline:
    def __init__(self, config: ViTTrainConfig) -> None:
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

        print(f"arch: vit_b_16")
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

        model = ViTFactory.build(self._c.num_classes)
        model.to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=self._c.lr,
            weight_decay=self._c.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self._c.epochs)

        runner = EpochRunner(model, self._device, criterion, optimizer)
        saver = CheckpointSaver(self._c.out_dir, "vit_b_16")

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
    cfg = ViTTrainConfig(
        data_root=root / "train",
        out_dir=Path("runs/trained"),
        num_classes=40,
        image_size=224,
        batch_size=32,
        epochs=30,
        lr=3e-4,
        weight_decay=0.05,
        num_workers=0,
        device="cpu",
    )
    ViTTrainingPipeline(cfg).run()


if __name__ == "__main__":
    main()