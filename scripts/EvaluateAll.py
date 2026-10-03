from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet34, vit_b_16
from ultralytics import YOLO


@dataclass
class EvalReport:
    model: str
    total: int = 0
    top1: int = 0
    top3: int = 0
    top5: int = 0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    per_class_precision: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_class_recall: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_class_f1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_class_support: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    macro_p: float = 0.0
    macro_r: float = 0.0
    macro_f1: float = 0.0
    weighted_p: float = 0.0
    weighted_r: float = 0.0
    weighted_f1: float = 0.0
    mean_conf: float = 0.0


class ChipDataset(Dataset):
    def __init__(self, root: Path, coco_json: Path, size: int = 128) -> None:
        self._root = root
        self._size = size

        coco = json.loads(coco_json.read_text())
        used_ids = {a["category_id"] for a in coco["annotations"]}
        self._coco_order = [
            c["name"] for c in sorted(
                [c for c in coco["categories"] if c["id"] in used_ids],
                key=lambda c: c["id"],
            )
        ]

        self._folder_classes = sorted(
            [d.name for d in root.iterdir() if d.is_dir()]
        )
        folder_to_idx = {c: i for i, c in enumerate(self._folder_classes)}

        self._items: list[tuple[Path, int]] = []
        for cls in self._folder_classes:
            for tif in sorted((root / cls).glob("*.tif")):
                self._items.append((tif, folder_to_idx[cls]))

        self._tf = transforms.Compose([
            transforms.Resize((size, size)),
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
        rgb = self._load_rgb(path)
        return self._tf(rgb), label

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as img:
            arr = np.array(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        uint8 = (arr * 255).astype(np.uint8)
        return Image.fromarray(np.stack([uint8] * 3, -1), mode="RGB")

    @property
    def classes(self) -> list[str]:
        return self._folder_classes

    @property
    def coco_order(self) -> list[str]:
        return self._coco_order

    @property
    def paths(self) -> list[Path]:
        return [p for p, _ in self._items]

    @property
    def labels(self) -> list[int]:
        return [y for _, y in self._items]


class ModelToFolderMap:
    def __init__(self, coco_order: list[str], folder_order: list[str]) -> None:
        folder_to_idx = {n: i for i, n in enumerate(folder_order)}
        self._map = np.array([folder_to_idx[n] for n in coco_order], dtype=np.int64)

    def apply(self, probs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(probs)
        for model_idx in range(probs.shape[1]):
            out[:, int(self._map[model_idx])] = probs[:, model_idx]
        return out

    @property
    def array(self) -> np.ndarray:
        return self._map


class TorchClassifier:
    def __init__(self, weights: Path, arch: str, num_classes: int) -> None:
        self._arch = arch
        self._model = self._build(arch, num_classes)
        self._load(weights)
        self._model.eval()

    def _build(self, arch: str, num_classes: int) -> nn.Module:
        if arch == "resnet18":
            m = resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        if arch == "resnet34":
            m = resnet34(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        if arch == "vit_b_16":
            m = vit_b_16(weights=None)
            m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
            return m
        raise ValueError(arch)

    def _load(self, weights: Path) -> None:
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        if self._arch == "vit_b_16":
            sd = {
                k.replace("heads.weight", "heads.head.weight")
                 .replace("heads.bias", "heads.head.bias"): v
                for k, v in sd.items()
            }
        self._model.load_state_dict(sd, strict=True)

    @property
    def name(self) -> str:
        return self._arch

    @torch.no_grad()
    def predict_probs(self, imgs: torch.Tensor) -> np.ndarray:
        logits = self._model(imgs)
        return torch.softmax(logits, dim=1).numpy()


class YoloClassifier:
    def __init__(self, weights: Path, num_classes: int, conf: float = 0.001) -> None:
        self._model = YOLO(str(weights))
        self._num_classes = num_classes
        self._conf = conf

    @property
    def name(self) -> str:
        return "yolo_v8"

    def predict_batch(self, paths: list[Path]) -> np.ndarray:
        results = self._model.predict(
            source=[str(p) for p in paths],
            imgsz=128,
            conf=self._conf,
            verbose=False,
            stream=True,
        )
        out = np.zeros((len(paths), self._num_classes), dtype=np.float32)
        for i, r in enumerate(results):
            for box in r.boxes:
                cls = int(box.cls.item())
                c = float(box.conf.item())
                if 0 <= cls < self._num_classes:
                    out[i, cls] += c
            s = out[i].sum()
            if s > 0:
                out[i] /= s
        return out


class MetricsAccumulator:
    def __init__(self, num_classes: int, model_map: ModelToFolderMap | None = None) -> None:
        self._num_classes = num_classes
        self._model_map = model_map
        self._cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self._top1 = self._top3 = self._top5 = self._total = 0
        self._conf_sum = 0.0

    def update(self, probs: np.ndarray, labels: np.ndarray) -> None:
        if self._model_map is not None:
            probs = self._model_map.apply(probs)
        preds = probs.argmax(axis=1)
        for i, (p, y) in enumerate(zip(preds, labels)):
            self._cm[int(y), int(p)] += 1
            self._total += 1
            self._conf_sum += float(probs[i, p])
            if int(p) == int(y):
                self._top1 += 1
            top = np.argsort(probs[i])[::-1]
            if int(y) in top[:3]:
                self._top3 += 1
            if int(y) in top[:5]:
                self._top5 += 1

    def compute(self, model_name: str) -> EvalReport:
        cm = self._cm
        support = cm.sum(axis=1)
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = support.astype(np.float64) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)
        w = support.astype(np.float64)
        wsum = w.sum() if w.sum() > 0 else 1.0
        return EvalReport(
            model=model_name,
            total=self._total,
            top1=self._top1,
            top3=self._top3,
            top5=self._top5,
            confusion=cm,
            per_class_precision=precision,
            per_class_recall=recall,
            per_class_f1=f1,
            per_class_support=support,
            macro_p=float(precision.mean()),
            macro_r=float(recall.mean()),
            macro_f1=float(f1.mean()),
            weighted_p=float((precision * w).sum() / wsum),
            weighted_r=float((recall * w).sum() / wsum),
            weighted_f1=float((f1 * w).sum() / wsum),
            mean_conf=self._conf_sum / max(self._total, 1),
        )


class TorchEvaluator:
    def __init__(self, clf: TorchClassifier, model_map: ModelToFolderMap, batch_size: int = 64) -> None:
        self._clf = clf
        self._map = model_map
        self._batch = batch_size

    def run(self, dataset: ChipDataset) -> EvalReport:
        loader = DataLoader(dataset, batch_size=self._batch, num_workers=0, shuffle=False)
        acc = MetricsAccumulator(len(dataset.classes), self._map)
        for imgs, labels in loader:
            acc.update(self._clf.predict_probs(imgs), labels.numpy())
        return acc.compute(self._clf.name)


class YoloEvaluator:
    def __init__(self, clf: YoloClassifier, model_map: ModelToFolderMap, batch_size: int = 64) -> None:
        self._clf = clf
        self._map = model_map
        self._batch = batch_size

    def run(self, dataset: ChipDataset) -> EvalReport:
        paths = dataset.paths
        labels = np.array(dataset.labels)
        acc = MetricsAccumulator(len(dataset.classes), self._map)
        for i in range(0, len(paths), self._batch):
            probs = self._clf.predict_batch(paths[i:i + self._batch])
            acc.update(probs, labels[i:i + self._batch])
        return acc.compute(self._clf.name)


class ReportPrinter:
    def print(self, r: EvalReport, classes: list[str]) -> None:
        n = max(r.total, 1)
        print(f"\n{'=' * 80}")
        print(f"MODEL: {r.model}")
        print(f"{'=' * 80}")
        print(f"total:           {r.total}")
        print(f"top1:            {r.top1 / n:.4f} ({r.top1})")
        print(f"top3:            {r.top3 / n:.4f} ({r.top3})")
        print(f"top5:            {r.top5 / n:.4f} ({r.top5})")
        print(f"macro P/R/F1:    {r.macro_p:.4f} / {r.macro_r:.4f} / {r.macro_f1:.4f}")
        print(f"weighted P/R/F1: {r.weighted_p:.4f} / {r.weighted_r:.4f} / {r.weighted_f1:.4f}")
        print(f"mean confidence: {r.mean_conf:.4f}")
        print(f"\nper-class (P / R / F1 / support):")
        for i, c in enumerate(classes):
            print(f"  {i:2d} {c:32s} "
                  f"{r.per_class_precision[i]:.3f} "
                  f"{r.per_class_recall[i]:.3f} "
                  f"{r.per_class_f1[i]:.3f} "
                  f"{int(r.per_class_support[i])}")
        errors = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and r.confusion[i, j] > 0:
                    errors.append((int(r.confusion[i, j]), classes[i], classes[j]))
        errors.sort(reverse=True)
        print(f"\ntop-10 ошибок (gt → pred, count):")
        for cnt, gt, pred in errors[:10]:
            print(f"  {cnt:4d}  {gt} → {pred}")


class ComparisonPrinter:
    def print(self, reports: list[EvalReport]) -> None:
        header = (
            f"{'model':<14} {'OA':>8} {'top3':>8} {'top5':>8} "
            f"{'macroP':>8} {'macroR':>8} {'macroF1':>8} "
            f"{'wF1':>8} {'conf':>8}"
        )
        print(f"\n{'=' * 100}")
        print("COMPARISON")
        print(f"{'=' * 100}")
        print(header)
        print("-" * len(header))
        for r in reports:
            n = max(r.total, 1)
            print(
                f"{r.model:<14} {r.top1 / n:>8.4f} {r.top3 / n:>8.4f} {r.top5 / n:>8.4f} "
                f"{r.macro_p:>8.4f} {r.macro_r:>8.4f} {r.macro_f1:>8.4f} "
                f"{r.weighted_f1:>8.4f} {r.mean_conf:>8.4f}"
            )


class ReportSaver:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, reports: list[EvalReport], classes: list[str]) -> None:
        summary = []
        for r in reports:
            n = max(r.total, 1)
            summary.append({
                "model": r.model,
                "total": r.total,
                "top1": r.top1 / n,
                "top3": r.top3 / n,
                "top5": r.top5 / n,
                "macro_p": r.macro_p,
                "macro_r": r.macro_r,
                "macro_f1": r.macro_f1,
                "weighted_p": r.weighted_p,
                "weighted_r": r.weighted_r,
                "weighted_f1": r.weighted_f1,
                "mean_conf": r.mean_conf,
                "per_class": [
                    {
                        "index": i, "name": c,
                        "precision": float(r.per_class_precision[i]),
                        "recall": float(r.per_class_recall[i]),
                        "f1": float(r.per_class_f1[i]),
                        "support": int(r.per_class_support[i]),
                    } for i, c in enumerate(classes)
                ],
            })
            np.savetxt(self._out_dir / f"confusion_{r.model}.csv",
                       r.confusion, fmt="%d", delimiter=",")
        (self._out_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        print(f"\nsaved -> {self._out_dir}")


class EvaluationPipeline:
    def __init__(
        self,
        test_root: Path,
        coco_json: Path,
        weights_root: Path,
        yolo_weights: Path,
        out_dir: Path,
    ) -> None:
        self._test_root = test_root
        self._coco_json = coco_json
        self._weights_root = weights_root
        self._yolo_weights = yolo_weights
        self._out_dir = out_dir

    def run(self) -> None:
        dataset = ChipDataset(self._test_root, self._coco_json)
        classes = dataset.classes
        model_map = ModelToFolderMap(dataset.coco_order, classes)

        print(f"test root: {self._test_root}")
        print(f"classes:   {len(classes)}")
        print(f"samples:   {len(dataset)}")
        print(f"map[:10]:  {model_map.array[:10]}")

        reports: list[EvalReport] = []

        for arch, subdir in [("resnet18", "ResNet18"), ("resnet34", "ResNet34")]:
            w = self._weights_root / subdir / "model/SOC_40classes.pth"
            if not w.exists():
                print(f"skip {arch}: no weights")
                continue
            print(f"\n>>> {arch} ...")
            clf = TorchClassifier(w, arch, len(classes))
            rep = TorchEvaluator(clf, model_map).run(dataset)
            reports.append(rep)
            ReportPrinter().print(rep, classes)

        vit_w = self._weights_root / "ViT/model/SOC_40classes.pth"
        if vit_w.exists():
            print(f"\n>>> vit_b_16 ...")
            try:
                clf = TorchClassifier(vit_w, "vit_b_16", len(classes))
                rep = TorchEvaluator(clf, model_map).run(dataset)
                reports.append(rep)
                ReportPrinter().print(rep, classes)
            except Exception as e:
                print(f"skip vit: {e}")

        if self._yolo_weights.exists():
            print(f"\n>>> yolo_v8 ...")
            clf = YoloClassifier(self._yolo_weights, len(classes))
            rep = YoloEvaluator(clf, model_map).run(dataset)
            reports.append(rep)
            ReportPrinter().print(rep, classes)

        ComparisonPrinter().print(reports)
        ReportSaver(self._out_dir).save(reports, classes)


def main() -> None:
    root = Path("data/_downloads")
    EvaluationPipeline(
        test_root=root / "Ground_Range/Amplitude_8bit/extracted/SOC_40classes/test",
        coco_json=root / "Ground_Range/Annotation_COCO/annotation_coco/SOC_40classes/annotations/train.json",
        weights_root=root / "ATRBench/Classification",
        yolo_weights=root / "ATRBench/Detection/unpacked/weight/v8/SOC_40classes/SOC_40classes_train/weights/best.pt",
        out_dir=Path("runs/soc40_eval"),
    ).run()


if __name__ == "__main__":
    main()