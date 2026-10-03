from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet34


class SOC40Dataset(Dataset):
    def __init__(self, root: Path, size: int = 128) -> None:
        self._root = root
        self._classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

        self._items: list[tuple[Path, int]] = []
        for cls in self._classes:
            for tif in sorted((root / cls).glob("*.tif")):
                self._items.append((tif, self._class_to_idx[cls]))

        self._tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        path, label = self._items[idx]
        with Image.open(path) as img:
            arr = np.array(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        uint8 = (arr * 255).astype(np.uint8)
        rgb = Image.fromarray(
            np.stack([uint8] * 3, -1), mode="RGB"
        )
        return self._tf(rgb), label

    @property
    def classes(self) -> list[str]:
        return self._classes


class ResNetEvaluator:
    def __init__(self, weights: Path, arch: str, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._arch = arch

        if arch == "resnet18":
            model = resnet18(weights=None)
        elif arch == "resnet34":
            model = resnet34(weights=None)
        else:
            raise ValueError(arch)

        model.fc = torch.nn.Linear(model.fc.in_features, 40)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.to(self._device).eval()
        self._model = model

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        total = 0
        correct = 0
        per_class_correct: dict[int, int] = {}
        per_class_total: dict[int, int] = {}

        for imgs, labels in loader:
            imgs = imgs.to(self._device)
            logits = self._model(imgs)
            preds = logits.argmax(dim=1).cpu()
            for pred, gt in zip(preds.tolist(), labels.tolist()):
                total += 1
                per_class_total[gt] = per_class_total.get(gt, 0) + 1
                if pred == gt:
                    correct += 1
                    per_class_correct[gt] = per_class_correct.get(gt, 0) + 1

        return {
            "model": self._arch,
            "total": total,
            "correct": correct,
            "oa": correct / max(total, 1),
            "per_class_total": per_class_total,
            "per_class_correct": per_class_correct,
        }


class EvaluationPipeline:
    def __init__(
        self,
        test_root: Path,
        out_dir: Path,
        num_workers: int = 4,
        batch_size: int = 64,
    ) -> None:
        self._test_root = test_root
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._num_workers = num_workers
        self._batch_size = batch_size

    def run(self) -> None:
        ds = SOC40Dataset(self._test_root)
        print(f"classes: {len(ds.classes)}")
        print(f"test samples: {len(ds)}")

        loader = DataLoader(
            ds,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

        weights_root = Path(
            "data/_downloads/ATRBench/Classification"
        )

        reports = []
        for arch, subdir in [("resnet18", "ResNet18"), ("resnet34", "ResNet34")]:
            w = weights_root / subdir / "model/SOC_40classes.pth"
            if not w.exists():
                print(f"skip {arch}: no weights at {w}")
                continue
            print(f"\n=== {arch} ===")
            evaluator = ResNetEvaluator(w, arch)
            report = evaluator.evaluate(loader)
            print(f"OA = {report['oa'] * 100:.2f}% ({report['correct']}/{report['total']})")
            reports.append(report)

        if reports:
            out = self._out_dir / "classifiers_soc40.json"
            out.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
            print(f"\nsaved -> {out}")


def main() -> None:
    EvaluationPipeline(
        test_root=Path(
            "data/_downloads/Ground_Range/Amplitude_8bit/"
            "extracted/SOC_40classes/test"
        ),
        out_dir=Path("runs/soc40_eval"),
    ).run()


if __name__ == "__main__":
    main()