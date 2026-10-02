from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


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


class SarChipExtractor:
    def __init__(self, size: int = 128) -> None:
        self._size = size

    def cut(self, image_path: Path, obj: GtObject, out_path: Path) -> None:
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

            cx = (obj.xmin + obj.xmax) // 2
            cy = (obj.ymin + obj.ymax) // 2
            half = self._size // 2
            left = max(0, min(rgb.width - self._size, cx - half))
            top = max(0, min(rgb.height - self._size, cy - half))
            rgb.crop(
                (left, top, left + self._size, top + self._size)
            ).save(out_path)


class IndexLoader:
    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path

    def load(self) -> dict[str, dict[str, str]]:
        return json.loads(self._index_path.read_text())


class SceneCollector:
    def __init__(
        self,
        base: Path,
        index: dict[str, dict[str, str]],
    ) -> None:
        self._base = base
        self._index = index

    def collect(self) -> list[SceneRef]:
        scenes: list[SceneRef] = []
        for folder, names in sorted(self._index.items()):
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


class MetricsComputer:
    def __init__(self, bundle: MetricsBundle) -> None:
        self._bundle = bundle

    def overall(self) -> dict[str, float]:
        preds = self._bundle.predictions
        total = len(preds)
        if total == 0:
            return {"total": 0, "correct": 0, "empty": 0, "top1": 0.0}
        correct = sum(1 for p in preds if p.gt == p.top1)
        empty = sum(1 for p in preds if p.top1 is None)
        return {
            "total": total,
            "correct": correct,
            "empty": empty,
            "top1": correct / total,
        }

    def per_class(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for p in self._bundle.predictions:
            s = stats.setdefault(p.gt, {"total": 0, "correct": 0, "recall": 0.0})
            s["total"] += 1
            if p.gt == p.top1:
                s["correct"] += 1
        for cls, s in stats.items():
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
        for sc, s in stats.items():
            if s["total"] > 0:
                s["acc"] = s["correct"] / s["total"]
        return stats

    def confusion(self) -> tuple[list[str], np.ndarray]:
        classes = self._bundle.classes
        idx = {c: i for i, c in enumerate(classes)}
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for p in self._bundle.predictions:
            if p.top1 is None:
                continue
            if p.gt not in idx or p.top1 not in idx:
                continue
            matrix[idx[p.gt], idx[p.top1]] += 1
        return classes, matrix


class MetricsPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def overall_bar(self, overall: dict[str, float]) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["correct", "wrong", "empty"]
        total = overall["total"]
        correct = overall["correct"]
        empty = overall["empty"]
        wrong = total - correct - empty
        values = [correct, wrong, empty]
        colors = ["#4caf50", "#f44336", "#9e9e9e"]
        bars = ax.bar(labels, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(val)}",
                ha="center",
                va="bottom",
            )
        ax.set_title(f"Overall (top1={overall['top1']:.3f}, n={int(total)})")
        ax.set_ylabel("objects")
        fig.tight_layout()
        fig.savefig(self._out_dir / "overall_bar.png", dpi=140)
        plt.close(fig)

    def per_class_recall(self, per_class: dict[str, dict[str, float]]) -> None:
        items = sorted(
            ((c, s) for c, s in per_class.items() if s["total"] >= 1),
            key=lambda x: (-x[1]["recall"], -x[1]["total"]),
        )
        if not items:
            return
        classes = [c for c, _ in items]
        recalls = [s["recall"] for _, s in items]
        totals = [s["total"] for _, s in items]

        fig, ax = plt.subplots(figsize=(10, max(4, len(classes) * 0.25)))
        y = np.arange(len(classes))
        bars = ax.barh(y, recalls, color="#3f51b5")
        for bar, t, r in zip(bars, totals, recalls):
            ax.text(
                min(r + 0.01, 1.0),
                bar.get_y() + bar.get_height() / 2,
                f"n={int(t)}",
                va="center",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(classes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("recall")
        ax.set_title("Per-class recall (top1)")
        fig.tight_layout()
        fig.savefig(self._out_dir / "per_class_recall.png", dpi=140)
        plt.close(fig)

    def per_scene_accuracy(self, per_scene: dict[str, dict[str, float]]) -> None:
        items = sorted(per_scene.items())
        scenes = [s for s, _ in items]
        accs = [v["acc"] for _, v in items]
        totals = [v["total"] for _, v in items]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(scenes, accs, color="#009688")
        for bar, t in zip(bars, totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={int(t)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title("Per-scene top1 accuracy")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(self._out_dir / "per_scene_accuracy.png", dpi=140)
        plt.close(fig)

    def confusion_matrix(self, classes: list[str], matrix: np.ndarray) -> None:
        if matrix.size == 0:
            return
        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.3),
                                        max(8, len(classes) * 0.3)))
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title("Confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(self._out_dir / "confusion_matrix.png", dpi=140)
        plt.close(fig)

    def confidence_hist(self, bundle: MetricsBundle) -> None:
        confs = [p.conf for p in bundle.predictions if p.top1 is not None]
        if not confs:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(confs, bins=20, color="#ff9800", edgecolor="black")
        ax.set_xlabel("top-1 confidence")
        ax.set_ylabel("count")
        ax.set_title(f"Confidence distribution (n={len(confs)})")
        fig.tight_layout()
        fig.savefig(self._out_dir / "confidence_hist.png", dpi=140)
        plt.close(fig)

    def gt_vs_pred_conf(self, bundle: MetricsBundle) -> None:
        correct = [p.conf for p in bundle.predictions if p.gt == p.top1]
        wrong = [p.conf for p in bundle.predictions if p.top1 is not None and p.gt != p.top1]
        if not correct and not wrong:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 21)
        ax.hist(correct, bins=bins, alpha=0.6, label="correct", color="#4caf50")
        ax.hist(wrong, bins=bins, alpha=0.6, label="wrong", color="#f44336")
        ax.set_xlabel("top-1 confidence")
        ax.set_ylabel("count")
        ax.set_title("Confidence: correct vs wrong")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self._out_dir / "confidence_correct_wrong.png", dpi=140)
        plt.close(fig)


class MetricsApp:
    def __init__(
        self,
        weights: Path,
        index_path: Path,
        base: Path,
        out_dir: Path,
    ) -> None:
        self._weights = weights
        self._index_path = index_path
        self._base = base
        self._out_dir = out_dir

    def run(self) -> None:
        index = IndexLoader(self._index_path).load()
        scenes = SceneCollector(self._base, index).collect()
        print(f"scenes: {len(scenes)}")
        if not scenes:
            return

        classifier = ChipClassifier(self._weights)
        runner = PredictionRunner(classifier, self._out_dir / "chips")
        bundle = runner.run(scenes)
        print(f"predictions: {len(bundle.predictions)}")

        computer = MetricsComputer(bundle)
        overall = computer.overall()
        per_class = computer.per_class()
        per_scene = computer.per_scene()
        classes, matrix = computer.confusion()

        self._out_dir.mkdir(parents=True, exist_ok=True)
        (self._out_dir / "metrics.json").write_text(json.dumps({
            "overall": overall,
            "per_class": per_class,
            "per_scene": per_scene,
        }, indent=2, ensure_ascii=False))

        plotter = MetricsPlotter(self._out_dir)
        plotter.overall_bar(overall)
        plotter.per_class_recall(per_class)
        plotter.per_scene_accuracy(per_scene)
        plotter.confusion_matrix(classes, matrix)
        plotter.confidence_hist(bundle)
        plotter.gt_vs_pred_conf(bundle)

        print(f"overall: {overall}")
        print(f"saved -> {self._out_dir}")


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

    def cut(self, image_path: Path, obj: GtObject, out_path: Path) -> None:
        rgb = self._cache.get(image_path)
        cx = (obj.xmin + obj.xmax) // 2
        cy = (obj.ymin + obj.ymax) // 2
        half = self._size // 2
        left = max(0, min(rgb.width - self._size, cx - half))
        top = max(0, min(rgb.height - self._size, cy - half))
        rgb.crop((left, top, left + self._size, top + self._size)).save(out_path)


class ChipClassifier:
    def __init__(self, weights: Path, conf: float = 0.0) -> None:
        self._model = YOLO(str(weights))
        self._conf = conf

    @property
    def names(self) -> dict[int, str]:
        return dict(self._model.names)

    def classify_batch(
        self, chip_paths: list[Path], imgsz: int = 128, batch: int = 64
    ) -> list[tuple[str | None, float]]:
        out: list[tuple[str | None, float]] = []
        results = self._model.predict(
            source=[str(p) for p in chip_paths],
            imgsz=imgsz,
            conf=self._conf,
            verbose=False,
            stream=True,
            batch=batch,
        )
        for r in results:
            best_cls, best_conf = None, 0.0
            for box in r.boxes:
                c = float(box.conf.item())
                if c > best_conf:
                    best_conf = c
                    best_cls = int(box.cls.item())
            if best_cls is None:
                out.append((None, 0.0))
            else:
                out.append((self.names.get(best_cls, str(best_cls)), best_conf))
        return out


class PredictionRunner:
    def __init__(self, classifier: ChipClassifier, chips_dir: Path) -> None:
        self._classifier = classifier
        self._cache = SarSceneCache()
        self._extractor = ChipExtractor(self._cache, 128)
        self._chips_dir = chips_dir

    def run(self, scenes: list[SceneRef]) -> MetricsBundle:
        self._chips_dir.mkdir(parents=True, exist_ok=True)
        bundle = MetricsBundle()
        seen_classes: set[str] = set()

        for si, scene in enumerate(scenes):
            print(f"[{si+1}/{len(scenes)}] {scene.folder}: loading tif...", flush=True)
            objects = XmlReader(scene.xml_path).read()
            chip_paths: list[Path] = []
            for i, obj in enumerate(objects):
                chip = self._chips_dir / f"{scene.folder}_{i:04d}.png"
                self._extractor.cut(scene.image_path, obj, chip)
                chip_paths.append(chip)
                seen_classes.add(obj.type_name)

            print(f"[{si+1}/{len(scenes)}] {scene.folder}: predict {len(chip_paths)} chips", flush=True)
            predictions = self._classifier.classify_batch(chip_paths)
            for obj, (top1, conf) in zip(objects, predictions):
                bundle.predictions.append(Prediction(
                    scene=scene.folder, gt=obj.type_name, top1=top1, conf=conf,
                ))
                if top1 is not None:
                    seen_classes.add(top1)
            print(f"[{si+1}/{len(scenes)}] {scene.folder}: done", flush=True)

        bundle.classes = sorted(seen_classes)
        return bundle


def main() -> None:
    weights = Path(
        "data/_bench/ATRBench/Detection/unpacked/weight/"
        "v8/SOC_40classes/SOC_40classes_train/weights/best.pt"
    )
    index_path = Path("runs/pretrained_inference/sandstone_index.json")
    base = Path("runs/pretrained_inference/scene/Raw_data/Subset_Sandstone")
    out_dir = Path("runs/pretrained_inference/metrics")

    MetricsApp(weights, index_path, base, out_dir).run()


if __name__ == "__main__":
    main()