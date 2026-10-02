from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


@dataclass
class GtObject:
    type_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class SceneRef:
    folder: str
    image_path: Path
    xml_path: Path


@dataclass
class ChipPrediction:
    gt: str
    chip: Image.Image
    top_k: list[tuple[str, float]]


class SceneIndexLoader:
    def __init__(self, path: Path, base: Path) -> None:
        self._path = path
        self._base = base

    def load(self, limit: int | None = None) -> list[SceneRef]:
        index = json.loads(self._path.read_text())
        scenes: list[SceneRef] = []
        for folder, names in sorted(index.items()):
            tif = names.get("tif")
            xml = names.get("xml")
            if not tif or not xml:
                continue
            img = self._base / "Result" / folder / tif
            xml_p = self._base / "Annotation" / folder / xml
            if not img.exists() or not xml_p.exists():
                continue
            scenes.append(SceneRef(folder, img, xml_p))
        return scenes[:limit] if limit is not None else scenes


class XmlReader:
    def __init__(self, xml_path: Path) -> None:
        self._xml_path = xml_path

    def read(self) -> list[GtObject]:
        root = ET.parse(self._xml_path).getroot()
        out: list[GtObject] = []
        for obj in root.findall("object"):
            t = obj.find("type")
            bb = obj.find("bndbox")
            if t is None or bb is None:
                continue
            out.append(GtObject(
                type_name=t.text,
                xmin=int(bb.find("xmin").text),
                ymin=int(bb.find("ymin").text),
                xmax=int(bb.find("xmax").text),
                ymax=int(bb.find("ymax").text),
            ))
        return out


class SarSceneCache:
    def __init__(self) -> None:
        self._cache: dict[Path, Image.Image] = {}

    def get(self, image_path: Path) -> Image.Image:
        cached = self._cache.get(image_path)
        if cached is not None:
            return cached
        with Image.open(image_path) as img:
            arr = np.array(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        uint8 = (arr * 255.0).astype(np.uint8)
        rgb = Image.fromarray(np.stack([uint8] * 3, axis=-1), mode="RGB")
        self._cache[image_path] = rgb
        return rgb


class ChipExtractor:
    def __init__(self, cache: SarSceneCache, size: int = 128) -> None:
        self._cache = cache
        self._size = size

    def cut(self, image_path: Path, obj: GtObject) -> Image.Image:
        rgb = self._cache.get(image_path)
        cx = (obj.xmin + obj.xmax) // 2
        cy = (obj.ymin + obj.ymax) // 2
        half = self._size // 2
        left = max(0, min(rgb.width - self._size, cx - half))
        top = max(0, min(rgb.height - self._size, cy - half))
        return rgb.crop((left, top, left + self._size, top + self._size))


class ResNetBuilder:
    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes

    def build(self) -> torch.nn.Module:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, self._num_classes)
        return model


class WeightLoader:
    def __init__(self, weights_path: Path) -> None:
        self._weights_path = weights_path

    def load_into(self, model: torch.nn.Module) -> None:
        sd = torch.load(self._weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)


class ClassNamesLoader:
    def __init__(self, coco_json: Path) -> None:
        self._coco_json = coco_json

    def load(self) -> list[str]:
        d = json.loads(self._coco_json.read_text())
        used_ids = {a["category_id"] for a in d["annotations"]}
        used = sorted(
            [c for c in d["categories"] if c["id"] in used_ids],
            key=lambda c: c["id"],
        )
        return [c["name"] for c in used]


class ChipClassifier:
    def __init__(
        self,
        weights: Path,
        class_names: list[str],
        device: str = "mps",
        batch: int = 64,
    ) -> None:
        self._device = torch.device(device if self._available(device) else "cpu")
        self._class_names = class_names
        self._batch = batch

        model = ResNetBuilder(len(class_names)).build()
        WeightLoader(weights).load_into(model)
        model.to(self._device).eval()
        self._model = model

        self._tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def _available(name: str) -> bool:
        if name == "mps":
            return torch.backends.mps.is_available()
        if name == "cuda":
            return torch.cuda.is_available()
        return True

    @torch.no_grad()
    def topk(self, chips: list[Image.Image], k: int = 5) -> list[list[tuple[str, float]]]:
        out: list[list[tuple[str, float]]] = []
        for i in range(0, len(chips), self._batch):
            chunk = chips[i:i + self._batch]
            batch = torch.stack([self._tf(c) for c in chunk]).to(self._device)
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idx = probs.topk(k, dim=1)
            for row_p, row_i in zip(top_probs.tolist(), top_idx.tolist()):
                out.append([
                    (self._class_names[i], float(p))
                    for i, p in zip(row_i, row_p)
                ])
        return out


class SceneCollector:
    def __init__(self, classifier: ChipClassifier, cache: SarSceneCache) -> None:
        self._classifier = classifier
        self._extractor = ChipExtractor(cache, 128)

    def collect(self, scenes: list[SceneRef], top_k: int = 5) -> list[ChipPrediction]:
        all_chips: list[Image.Image] = []
        all_gts: list[str] = []

        for scene in scenes:
            objects = XmlReader(scene.xml_path).read()
            for obj in objects:
                all_chips.append(self._extractor.cut(scene.image_path, obj))
                all_gts.append(obj.type_name)

        topk = self._classifier.topk(all_chips, k=top_k)
        return [
            ChipPrediction(gt=g, chip=c, top_k=t)
            for g, c, t in zip(all_gts, all_chips, topk)
        ]


class GridPlotter:
    def __init__(self, out_path: Path, cols: int = 8) -> None:
        self._out_path = out_path
        self._cols = cols
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, preds: list[ChipPrediction]) -> None:
        n = len(preds)
        rows = (n + self._cols - 1) // self._cols
        fig, axes = plt.subplots(rows, self._cols, figsize=(self._cols * 2.0, rows * 2.6))
        axes = np.array(axes).reshape(-1) if rows * self._cols > 1 else [axes]

        for ax, p in zip(axes, preds):
            ax.imshow(p.chip)
            ax.axis("off")
            correct = p.top_k[0][0] == p.gt
            color = "#4caf50" if correct else "#f44336"
            top1 = p.top_k[0]
            ax.set_title(
                f"GT: {p.gt}\npred: {top1[0]} {top1[1]:.2f}",
                fontsize=8,
                color=color,
            )

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(self._out_path, dpi=140)
        plt.close(fig)
        print(f"saved -> {self._out_path}")


class ErrorGridPlotter:
    def __init__(self, out_path: Path, cols: int = 8) -> None:
        self._out_path = out_path
        self._cols = cols
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, preds: list[ChipPrediction], max_n: int = 64) -> None:
        wrong = [p for p in preds if p.top_k[0][0] != p.gt][:max_n]
        if not wrong:
            print("нет ошибок")
            return

        rows = (len(wrong) + self._cols - 1) // self._cols
        fig, axes = plt.subplots(rows, self._cols, figsize=(self._cols * 2.0, rows * 2.6))
        axes = np.array(axes).reshape(-1) if rows * self._cols > 1 else [axes]

        for ax, p in zip(axes, wrong):
            ax.imshow(p.chip)
            ax.axis("off")
            top3 = "\n".join(f"{n} {c:.2f}" for n, c in p.top_k[:3])
            ax.set_title(f"GT: {p.gt}\n{top3}", fontsize=7, color="#f44336")

        for ax in axes[len(wrong):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(self._out_path, dpi=140)
        plt.close(fig)
        print(f"saved -> {self._out_path}")


class ConfusionPlotter:
    def __init__(self, out_path: Path) -> None:
        self._out_path = out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def plot(self, preds: list[ChipPrediction], classes: list[str]) -> None:
        idx = {c: i for i, c in enumerate(classes)}
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for p in preds:
            if p.gt not in idx:
                continue
            pred = p.top_k[0][0]
            if pred not in idx:
                continue
            matrix[idx[p.gt], idx[pred]] += 1

        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title("ResNet18 confusion matrix (Sandstone)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(self._out_path, dpi=140)
        plt.close(fig)
        print(f"saved -> {self._out_path}")


class VisualizeApp:
    def __init__(
        self,
        weights: Path,
        coco_json: Path,
        index_path: Path,
        base: Path,
        out_dir: Path,
        device: str = "mps",
        scene_limit: int | None = None,
    ) -> None:
        self._weights = weights
        self._coco_json = coco_json
        self._index_path = index_path
        self._base = base
        self._out_dir = out_dir
        self._device = device
        self._scene_limit = scene_limit

    def run(self) -> None:
        scenes = SceneIndexLoader(self._index_path, self._base).load(self._scene_limit)
        print(f"scenes: {len(scenes)}")

        classes = ClassNamesLoader(self._coco_json).load()
        classifier = ChipClassifier(self._weights, classes, device=self._device)
        cache = SarSceneCache()

        preds = SceneCollector(classifier, cache).collect(scenes, top_k=5)
        print(f"chips: {len(preds)}")

        correct = sum(1 for p in preds if p.top_k[0][0] == p.gt)
        print(f"top1: {correct / max(len(preds), 1):.3f}")

        self._out_dir.mkdir(parents=True, exist_ok=True)
        GridPlotter(self._out_dir / "grid_all.png").plot(preds)
        ErrorGridPlotter(self._out_dir / "grid_errors.png").plot(preds, max_n=64)
        ConfusionPlotter(self._out_dir / "confusion.png").plot(preds, classes)


def main() -> None:
    weights = Path(
        "data/_bench/ATRBench/Classification/ResNet18/model/SOC_40classes.pth"
    )
    coco_json = Path(
        "data/_annot/Ground_Range/coco_outer/annotation_coco/"
        "SOC_40classes/annotations/train.json"
    )
    index_path = Path("runs/pretrained_inference/sandstone_index.json")
    base = Path("runs/pretrained_inference/scene/Raw_data/Subset_Sandstone")
    out_dir = Path("runs/resnet_eval/visuals")

    VisualizeApp(
        weights=weights,
        coco_json=coco_json,
        index_path=index_path,
        base=base,
        out_dir=out_dir,
        device="mps",
        scene_limit=None,
    ).run()


if __name__ == "__main__":
    main()