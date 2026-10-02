from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

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
class Prediction:
    scene: str
    gt: str
    top1: str | None
    conf: float


@dataclass
class MetricsBundle:
    predictions: list[Prediction] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


class SceneIndexLoader:
    def __init__(self, path: Path, base: Path) -> None:
        self._path = path
        self._base = base

    def load(self) -> list[SceneRef]:
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
        return scenes


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
    def __init__(self, num_classes: int = 40) -> None:
        self._num_classes = num_classes

    def build(self) -> torch.nn.Module:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, self._num_classes)
        return model


class WeightLoader:
    def __init__(self, weights_path: Path) -> None:
        self._weights_path = weights_path

    def strip_prefix(self, sd: dict) -> dict:
        return {k.replace("model.", "", 1): v for k, v in sd.items()}

    def load_into(self, model: torch.nn.Module) -> None:
        sd = torch.load(self._weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = self.strip_prefix(sd)
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

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @torch.no_grad()
    def classify(self, chips: list[Image.Image]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for i in range(0, len(chips), self._batch):
            chunk = chips[i:i + self._batch]
            batch = torch.stack([self._tf(c) for c in chunk]).to(self._device)
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)
            top_conf, top_idx = probs.max(dim=1)
            for cls_i, conf in zip(top_idx.tolist(), top_conf.tolist()):
                out.append((self._class_names[cls_i], float(conf)))
        return out


class PredictionRunner:
    def __init__(self, classifier: ChipClassifier) -> None:
        self._classifier = classifier
        self._cache = SarSceneCache()
        self._extractor = ChipExtractor(self._cache, 128)

    def run(self, scenes: list[SceneRef]) -> MetricsBundle:
        bundle = MetricsBundle()
        seen: set[str] = set()

        for si, scene in enumerate(scenes):
            objects = XmlReader(scene.xml_path).read()
            print(f"[{si+1}/{len(scenes)}] {scene.folder}: {len(objects)} objects", flush=True)

            chips = [self._extractor.cut(scene.image_path, o) for o in objects]
            preds = self._classifier.classify(chips)

            for obj, (top1, conf) in zip(objects, preds):
                bundle.predictions.append(Prediction(
                    scene=scene.folder,
                    gt=obj.type_name,
                    top1=top1,
                    conf=conf,
                ))
                seen.add(obj.type_name)
                seen.add(top1)

        bundle.classes = sorted(seen)
        return bundle


class MetricsComputer:
    def __init__(self, bundle: MetricsBundle) -> None:
        self._bundle = bundle

    def overall(self) -> dict[str, float]:
        preds = self._bundle.predictions
        total = len(preds)
        if total == 0:
            return {"total": 0, "correct": 0, "top1": 0.0}
        correct = sum(1 for p in preds if p.gt == p.top1)
        return {"total": total, "correct": correct, "top1": correct / total}

    def per_class(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for p in self._bundle.predictions:
            s = stats.setdefault(p.gt, {"total": 0, "correct": 0, "recall": 0.0})
            s["total"] += 1
            if p.gt == p.top1:
                s["correct"] += 1
        for s in stats.values():
            if s["total"] > 0:
                s["recall"] = s["correct"] / s["total"]
        return stats

    def per_scene(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for p in self._bundle.predictions:
            s = stats.setdefault(p.scene, {"total": 0, "correct": 0, "acc": 0.0})
            s["total"] += 1
            if p.gt == p.top1:
                s["correct"] += 1
        for s in stats.values():
            if s["total"] > 0:
                s["acc"] = s["correct"] / s["total"]
        return stats


class MetricsReporter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, bundle: MetricsBundle) -> None:
        computer = MetricsComputer(bundle)
        report = {
            "overall": computer.overall(),
            "per_class": computer.per_class(),
            "per_scene": computer.per_scene(),
        }
        (self._out_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        print(json.dumps(report["overall"], indent=2))
        print(f"saved -> {self._out_dir / 'metrics.json'}")


class ResNetEvalApp:
    def __init__(
        self,
        weights: Path,
        coco_json: Path,
        index_path: Path,
        base: Path,
        out_dir: Path,
        device: str = "mps",
    ) -> None:
        self._weights = weights
        self._coco_json = coco_json
        self._index_path = index_path
        self._base = base
        self._out_dir = out_dir
        self._device = device

    def run(self) -> None:
        scenes = SceneIndexLoader(self._index_path, self._base).load()
        print(f"scenes: {len(scenes)}")
        if not scenes:
            return

        classes = ClassNamesLoader(self._coco_json).load()
        print(f"classes: {len(classes)}")

        classifier = ChipClassifier(self._weights, classes, device=self._device)
        bundle = PredictionRunner(classifier).run(scenes)
        print(f"predictions: {len(bundle.predictions)}")
        MetricsReporter(self._out_dir).save(bundle)


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
    out_dir = Path("runs/resnet_eval")

    ResNetEvalApp(
        weights=weights,
        coco_json=coco_json,
        index_path=index_path,
        base=base,
        out_dir=out_dir,
        device="mps",
    ).run()


if __name__ == "__main__":
    main()