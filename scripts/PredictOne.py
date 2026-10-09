from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet34, vit_b_16
from ultralytics import YOLO

from datetime import datetime
        
@dataclass
class Prediction:
    model: str
    top_k: list[tuple[str, float]]



class ResultSaver:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self,
        image_path: Path,
        true_class: str,
        results: dict[str, list[tuple[str, float]]],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._out_dir / f"{image_path.stem}_{timestamp}.json"
        payload = {
            "image": str(image_path),
            "true_class": true_class,
            "predictions": {
                model: [
                    {"class": cls, "confidence": conf}
                    for cls, conf in preds
                ]
                for model, preds in results.items()
            },
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return out

    def save_txt(
        self,
        image_path: Path,
        true_class: str,
        results: dict[str, list[tuple[str, float]]],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._out_dir / f"{image_path.stem}_{timestamp}.txt"
        lines = [
            f"image: {image_path}",
            f"true_class: {true_class}",
            "",
        ]
        for model, preds in results.items():
            lines.append(f"[{model}]")
            for rank, (cls, conf) in enumerate(preds, 1):
                mark = "OK " if cls == true_class else "ERR"
                lines.append(f"  {mark} {rank}. {cls:35s} {conf:.4f}")
            lines.append("")
        out.write_text("\n".join(lines))
        return out

    def save_figure(
        self,
        image_path: Path,
        true_class: str,
        results: dict[str, list[tuple[str, float]]],
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._out_dir / f"{image_path.stem}_{timestamp}.png"

        with Image.open(image_path) as im:
            arr = np.array(im).astype(np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        arr8 = (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))
        axes = np.atleast_1d(axes)

        axes[0].imshow(arr8, cmap="gray")
        axes[0].set_title(f"input\ntrue: {true_class}", fontsize=9)
        axes[0].axis("off")

        for ax, (model, preds) in zip(axes[1:], results.items()):
            names = [c for c, _ in preds][::-1]
            confs = [p for _, p in preds][::-1]
            colors = ["#4caf50" if n == true_class else "#f44336" for n in names]
            ax.barh(np.arange(len(names)), confs, color=colors)
            ax.set_yticks(np.arange(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_title(model, fontsize=9)

        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out

class BoxVisualizer:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import ImageFont
        try:
            self._font = ImageFont.truetype("DejaVuSans.ttf", 11)
        except OSError:
            self._font = ImageFont.load_default()

    def save_classifier(
        self,
        image_path: Path,
        image: Image.Image,
        true_class: str,
        model_name: str,
        preds: list[tuple[str, float]],
    ) -> Path:
        from PIL import ImageDraw

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._out_dir / f"{image_path.stem}_{timestamp}_{model_name}.png"

        canvas = image.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)
        W, H = canvas.size

        top1, conf = preds[0] if preds else ("<none>", 0.0)
        correct = top1 == true_class
        color = (0, 200, 0) if correct else (220, 0, 0)

        draw.rectangle([2, 2, W - 3, H - 3], outline=color, width=2)

        lines = [
            f"pred: {top1} ({conf:.2f})",
            f"true: {true_class}",
        ]
        y = 4
        for line in lines:
            draw.text((6, y), line, fill=color, font=self._font)
            y += 14

        canvas.save(out)
        return out

    def save_yolo(
        self,
        image_path: Path,
        image: Image.Image,
        true_class: str,
        model_name: str,
        boxes: list[tuple[int, int, int, int, str, float]],
    ) -> Path:
        from PIL import ImageDraw

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._out_dir / f"{image_path.stem}_{timestamp}_{model_name}.png"

        canvas = image.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)

        for x1, y1, x2, y2, name, conf in boxes:
            correct = name == true_class
            color = (0, 200, 0) if correct else (220, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text(
                (x1, max(0, y1 - 12)),
                f"{name} {conf:.2f}",
                fill=color,
                font=self._font,
            )

        if not boxes:
            draw.text((6, 6), "no detections", fill=(220, 0, 0), font=self._font)

        canvas.save(out)
        return out

    
class PredictOneApp:
    def __init__(
        self,
        image_path: Path,
        weights_root: Path,
        yolo_weights: Path,
        coco_json: Path,
        out_dir: Path,
    ) -> None:
        self._image_path = image_path
        self._weights_root = weights_root
        self._yolo_weights = yolo_weights
        self._coco_json = coco_json
        self._out_dir = out_dir
        self._pre = SarPreprocessor()
        self._saver = ResultSaver(out_dir)
        self._visualizer = BoxVisualizer(out_dir)

    def run(self) -> None:
        true_class = self._image_path.parent.name
        print(f"image: {self._image_path}")
        print(f"true class (folder): {true_class}")

        class_names = ClassNamesLoader(self._coco_json).load()
        print(f"classes: {len(class_names)}")

        image = self._pre.to_rgb(self._image_path)
        print(f"image size: {image.size}")

        models = ModelRegistry(
            self._weights_root, self._yolo_weights, class_names
        ).build()
        print(f"models loaded: {[m.name for m in models]}")

        printer = PredictionPrinter()
        results: dict[str, list[tuple[str, float]]] = {}
        for m in models:
            preds = m.predict(image, k=5)
            printer.print(m.name, true_class, preds)
            results[m.name] = preds

        json_path = self._saver.save_json(self._image_path, true_class, results)
        txt_path = self._saver.save_txt(self._image_path, true_class, results)

        figure_paths: list[Path] = []
        for m in models:
            if isinstance(m, YoloClassifier):
                boxes = m.detect_boxes(image)
                p = self._visualizer.save_yolo(
                    self._image_path, image, true_class, m.name, boxes
                )
            else:
                p = self._visualizer.save_classifier(
                    self._image_path, image, true_class, m.name, results[m.name]
                )
            figure_paths.append(p)

        print(f"\nsaved:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")
        for p in figure_paths:
            print(f"  IMG:  {p}")


class ClassNamesLoader:
    def __init__(self, coco_json: Path) -> None:
        self._coco_json = coco_json

    def load(self) -> list[str]:
        d = json.loads(self._coco_json.read_text())
        used_ids = {a["category_id"] for a in d["annotations"]}
        cats = sorted(
            [c for c in d["categories"] if c["id"] in used_ids],
            key=lambda c: c["id"],
        )
        return [c["name"] for c in cats]


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


class TorchClassifier:
    def __init__(
        self,
        weights: Path,
        arch: str,
        class_names: list[str],
        device: str = "cpu",
    ) -> None:
        self._arch = arch
        self._class_names = class_names
        self._device = torch.device(device)

        self._model = self._build(arch, len(class_names))
        self._load(weights)
        self._model.to(self._device).eval()

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

    @torch.no_grad()
    def predict(self, image: Image.Image, k: int = 5) -> list[tuple[str, float]]:
        tensor = self._tf(image).unsqueeze(0).to(self._device)
        logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k)
        return [
            (self._class_names[i], float(p))
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]


class YoloClassifier:
    def __init__(
        self,
        weights: Path,
        class_names: list[str],
        conf: float = 0.001,
        imgsz: int = 128,
    ) -> None:
        self._class_names = class_names
        self._model = YOLO(str(weights))
        self._conf = conf
        self._imgsz = imgsz

    @property
    def name(self) -> str:
        return "yolo_v8"

    def detect_boxes(
        self, image: Image.Image
    ) -> list[tuple[int, int, int, int, str, float]]:
        results = self._model.predict(
            source=np.array(image),
            imgsz=self._imgsz,
            conf=self._conf,
            verbose=False,
        )
        out: list[tuple[int, int, int, int, str, float]] = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                name = (
                    self._class_names[cls]
                    if cls < len(self._class_names)
                    else f"class_{cls}"
                )
                out.append((
                    int(xyxy[0]), int(xyxy[1]),
                    int(xyxy[2]), int(xyxy[3]),
                    name, conf,
                ))
        return out

    def predict(self, image: Image.Image, k: int = 5) -> list[tuple[str, float]]:
        boxes = self.detect_boxes(image)
        cls_to_score: dict[str, float] = {}
        for _, _, _, _, name, conf in boxes:
            if name not in cls_to_score or conf > cls_to_score[name]:
                cls_to_score[name] = conf
        if not cls_to_score:
            return [("<no detection>", 0.0)]
        ranked = sorted(cls_to_score.items(), key=lambda x: -x[1])[:k]
        return ranked


class ModelRegistry:
    def __init__(
        self,
        weights_root: Path,
        yolo_weights: Path,
        class_names: list[str],
    ) -> None:
        self._weights_root = weights_root
        self._yolo_weights = yolo_weights
        self._class_names = class_names

    def build(self) -> list[object]:
        models: list[object] = []

        for arch, subdir in [("resnet18", "ResNet18"), ("resnet34", "ResNet34")]:
            w = self._weights_root / subdir / "model/SOC_40classes.pth"
            if w.exists():
                models.append(TorchClassifier(w, arch, self._class_names))

        vit_w = self._weights_root / "ViT/model/SOC_40classes.pth"
        if vit_w.exists():
            models.append(TorchClassifier(vit_w, "vit_b_16", self._class_names))

        if self._yolo_weights.exists():
            models.append(YoloClassifier(self._yolo_weights, self._class_names))

        return models


class PredictionPrinter:
    def print(self, model_name: str, true_class: str, preds: list[tuple[str, float]]) -> None:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")
        print(f"true class: {true_class}")
        print(f"top-{len(preds)}:")
        for rank, (cls, conf) in enumerate(preds, 1):
            mark = "✓" if cls == true_class else " "
            print(f"  {mark} {rank}. {cls:35s} {conf:.4f}")



def main() -> None:
    root = Path("data/_downloads")
    test_root = root / "Ground_Range/Amplitude_8bit/extracted/SOC_40classes/test"

    image_path = next((test_root / "Buick_Excelle_GT").glob("*.tif"))
    print(f"resolved: {image_path}")

    PredictOneApp(
        image_path=image_path,
        weights_root=root / "ATRBench/Classification",
        yolo_weights=root / "ATRBench/Detection/unpacked/weight/v8/SOC_40classes/SOC_40classes_train/weights/best.pt",
        coco_json=root / "Ground_Range/Annotation_COCO/annotation_coco/SOC_40classes/annotations/train.json",
        out_dir=Path("runs/predict_one"),
    ).run()


if __name__ == "__main__":
    main()