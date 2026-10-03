from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import resnet18, resnet34, vit_b_16
from ultralytics import YOLO


@dataclass
class Box:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    score: float = 1.0
    label: str = ""


@dataclass
class Scene:
    folder: str
    image_path: Path
    xml_path: Path


class SceneIndex:
    def __init__(self, index_path: Path, base: Path) -> None:
        self._index = json.loads(index_path.read_text())
        self._base = base

    def load(self) -> list[Scene]:
        out: list[Scene] = []
        for folder, names in sorted(self._index.items()):
            tif = self._base / "Result" / folder / names["tif"]
            xml = self._base / "Annotation" / folder / names["xml"]
            if tif.exists() and xml.exists():
                out.append(Scene(folder, tif, xml))
        return out


class XmlGtReader:
    def __init__(self, xml_path: Path) -> None:
        self._xml_path = xml_path

    def read(self) -> list[Box]:
        root = ET.parse(self._xml_path).getroot()
        out: list[Box] = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            t = obj.find("type")
            if bb is None:
                continue
            out.append(Box(
                xmin=int(bb.find("xmin").text),
                ymin=int(bb.find("ymin").text),
                xmax=int(bb.find("xmax").text),
                ymax=int(bb.find("ymax").text),
                score=1.0,
                label=(t.text if t is not None else "vehicle"),
            ))
        return out


class SceneLoader:
    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        self._low = low
        self._high = high

    def load_rgb(self, path: Path) -> Image.Image:
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


class YoloSceneDetector:
    def __init__(self, weights: Path, conf: float = 0.1) -> None:
        self._model = YOLO(str(weights))
        self._conf = conf

    @property
    def name(self) -> str:
        return "yolo_v8"

    def detect(self, image: Image.Image, imgsz: int = 128) -> list[Box]:
        results = self._model.predict(
            source=np.array(image),
            imgsz=imgsz,
            conf=self._conf,
            verbose=False,
        )
        out: list[Box] = []
        for r in results:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                out.append(Box(
                    xmin=int(xyxy[0]), ymin=int(xyxy[1]),
                    xmax=int(xyxy[2]), ymax=int(xyxy[3]),
                    score=float(b.conf.item()),
                    label=self._model.names[int(b.cls.item())],
                ))
        return out


class ClassifierSlidingDetector:
    def __init__(
        self,
        weights: Path,
        arch: str,
        num_classes: int,
        window: int = 128,
        stride: int = 64,
        conf_threshold: float = 0.5,
    ) -> None:
        self._arch = arch
        self._window = window
        self._stride = stride
        self._conf_threshold = conf_threshold

        self._model = self._build(arch, num_classes)
        self._load(weights)
        self._model.eval()

        self._tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

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

    def detect(self, image: Image.Image) -> list[Box]:
        W, H = image.size
        w, h = self._window, self._window
        wins: list[tuple[int, int]] = []
        for y in range(0, H - h + 1, self._stride):
            for x in range(0, W - w + 1, self._stride):
                wins.append((x, y))
        if not wins:
            return []

        # батчами
        batch_size = 64
        kept: list[Box] = []
        for i in range(0, len(wins), batch_size):
            chunk = wins[i:i + batch_size]
            tensors = []
            for (x, y) in chunk:
                crop = image.crop((x, y, x + w, y + h))
                tensors.append(self._tf(crop))
            batch = torch.stack(tensors)
            with torch.no_grad():
                probs = torch.softmax(self._model(batch), dim=1)
                confs, idxs = probs.max(dim=1)
            for (x, y), c, k in zip(chunk, confs.tolist(), idxs.tolist()):
                if c >= self._conf_threshold:
                    kept.append(Box(
                        xmin=x, ymin=y, xmax=x + w, ymax=y + h,
                        score=float(c),
                        label="vehicle",
                    ))
        return self._nms(kept, iou_threshold=0.3)

    @staticmethod
    def _nms(boxes: list[Box], iou_threshold: float = 0.3) -> list[Box]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: -b.score)
        keep: list[Box] = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if _iou(best, b) < iou_threshold]
        return keep


def _iou(a: Box, b: Box) -> float:
    x1 = max(a.xmin, b.xmin)
    y1 = max(a.ymin, b.ymin)
    x2 = min(a.xmax, b.xmax)
    y2 = min(a.ymax, b.ymax)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SceneRenderer:
    def __init__(self, max_side: int = 2048) -> None:
        self._max_side = max_side

    def scale(self, image: Image.Image) -> tuple[Image.Image, float]:
        W, H = image.size
        s = self._max_side / max(W, H)
        if s >= 1.0:
            return image, 1.0
        return image.resize((int(W * s), int(H * s)), Image.LANCZOS), s

    def draw(
        self,
        image: Image.Image,
        gt: list[Box],
        pred: list[Box],
        scale: float,
    ) -> Image.Image:
        canvas = image.copy()
        d = ImageDraw.Draw(canvas)
        for b in gt:
            d.rectangle(
                [b.xmin * scale, b.ymin * scale, b.xmax * scale, b.ymax * scale],
                outline=(0, 255, 0), width=2,
            )
        for b in pred:
            d.rectangle(
                [b.xmin * scale, b.ymin * scale, b.xmax * scale, b.ymax * scale],
                outline=(255, 64, 64), width=2,
            )
            d.text(
                (b.xmin * scale, b.ymin * scale),
                f"{b.label} {b.score:.2f}",
                fill=(255, 64, 64),
            )
        return canvas


class DetectionPipeline:
    def __init__(
        self,
        weights_root: Path,
        yolo_weights: Path,
        out_dir: Path,
        conf_classifier: float = 0.5,
        conf_yolo: float = 0.1,
    ) -> None:
        self._weights_root = weights_root
        self._yolo_weights = yolo_weights
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._loader = SceneLoader()
        self._renderer = SceneRenderer()
        self._conf_classifier = conf_classifier
        self._conf_yolo = conf_yolo

    def _build_detectors(self, num_classes: int) -> list[object]:
        models: list[object] = []
        for arch, subdir in [("resnet18", "ResNet18"), ("resnet34", "ResNet34")]:
            w = self._weights_root / subdir / "model/SOC_40classes.pth"
            if w.exists():
                models.append(ClassifierSlidingDetector(
                    w, arch, num_classes,
                    conf_threshold=self._conf_classifier,
                ))
                print(f"loaded {arch}")
        vit_w = self._weights_root / "ViT/model/SOC_40classes.pth"
        if vit_w.exists():
            try:
                models.append(ClassifierSlidingDetector(
                    vit_w, "vit_b_16", num_classes,
                    conf_threshold=self._conf_classifier,
                ))
                print("loaded vit_b_16")
            except Exception as e:
                print(f"skip vit: {e}")
        if self._yolo_weights.exists():
            models.append(YoloSceneDetector(self._yolo_weights, conf=self._conf_yolo))
            print("loaded yolo_v8")
        return models

    def run(self, scenes: list[Scene], num_classes: int = 40) -> None:
        detectors = self._build_detectors(num_classes)

        for scene in scenes:
            print(f"\n=== {scene.folder} ===")
            image = self._loader.load_rgb(scene.image_path)
            gt = XmlGtReader(scene.xml_path).read()
            scaled, s = self._renderer.scale(image)

            for det in detectors:
                print(f"  {det.name} ...", flush=True)
                preds = det.detect(image)
                print(f"    found: {len(preds)}, gt: {len(gt)}")

                canvas = self._renderer.draw(scaled, gt, preds, s)
                out = self._out_dir / f"{scene.folder}_{det.name}.png"
                canvas.save(out)


def main() -> None:
    root = Path("data/_downloads")
    pipeline = DetectionPipeline(
        weights_root=root / "ATRBench/Classification",
        yolo_weights=root / "ATRBench/Detection/unpacked/weight/v8/SOC_40classes/SOC_40classes_train/weights/best.pt",
        out_dir=Path("runs/detections"),
        conf_classifier=0.5,
        conf_yolo=0.1,
    )
    scenes = SceneIndex(
        Path("data/sandstone_index.json"),
        Path("data/sandstone/Raw_data/Subset_Sandstone"),
    ).load()
    print(f"scenes: {len(scenes)}")
    pipeline.run(scenes, num_classes=40)


if __name__ == "__main__":
    main()