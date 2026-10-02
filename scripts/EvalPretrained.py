from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

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
    image_path: Path
    xml_path: Path


@dataclass
class Prediction:
    gt: str
    top1: str | None
    conf: float


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


class ChipExtractor:
    def __init__(self, image_path: Path, size: int = 128) -> None:
        self._image_path = image_path
        self._size = size

    def cut(self, obj: GtObject, out_path: Path) -> None:
        with Image.open(self._image_path) as img:
            arr = np.array(img).astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            uint8 = (arr * 255.0).astype(np.uint8)
            rgb = Image.fromarray(np.stack([uint8] * 3, axis=-1), mode="RGB")

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

    def classify(self, chip_path: Path, imgsz: int = 128) -> tuple[str | None, float]:
        results = self._model.predict(
            source=str(chip_path), imgsz=imgsz, conf=self._conf, verbose=False
        )
        best_cls, best_conf = None, 0.0
        for r in results:
            for box in r.boxes:
                c = float(box.conf.item())
                if c > best_conf:
                    best_conf = c
                    best_cls = int(box.cls.item())
        if best_cls is None:
            return None, 0.0
        return self.names.get(best_cls, str(best_cls)), best_conf


class Evaluator:
    def __init__(self, classifier: ChipClassifier, work_dir: Path) -> None:
        self._clf = classifier
        self._work_dir = work_dir
        self._total = 0
        self._correct = 0
        self._empty = 0
        self._rows: list[Prediction] = []

    def run(self, scene: SceneRef) -> None:
        objects = XmlReader(scene.xml_path).read()
        extractor = ChipExtractor(scene.image_path, 128)
        chips_dir = self._work_dir / "gt_chips"
        chips_dir.mkdir(parents=True, exist_ok=True)

        for i, obj in enumerate(objects):
            chip = chips_dir / f"chip_{i:04d}.png"
            extractor.cut(obj, chip)
            top1, conf = self._clf.classify(chip)
            self._total += 1
            if top1 is None:
                self._empty += 1
            elif top1 == obj.type_name:
                self._correct += 1
            self._rows.append(Prediction(obj.type_name, top1, conf))

    def report(self) -> None:
        for r in self._rows:
            mark = "OK " if r.gt == r.top1 else "ERR"
            print(f"{mark} gt={r.gt:30s} pred={r.top1 or '-':30s} conf={r.conf:.3f}")
        print(f"\ntotal={self._total} correct={self._correct} empty={self._empty}")
        if self._total:
            print(f"top1_accuracy={self._correct / self._total:.3f}")


class EvalApp:
    def __init__(self, weights: Path, work_dir: Path) -> None:
        self._clf = ChipClassifier(weights)
        self._work_dir = work_dir

    def run(self, image_path: Path, xml_path: Path) -> None:
        scene = SceneRef(image_path, xml_path)
        ev = Evaluator(self._clf, self._work_dir)
        ev.run(scene)
        ev.report()


def main() -> None:
    weights = Path(
        "data/_bench/ATRBench/Detection/unpacked/weight/"
        "v8/SOC_40classes/SOC_40classes_train/weights/best.pt"
    )
    base = Path("runs/pretrained_inference/scene/Raw_data/Subset_Sandstone")
    image_path = base / "Result/30deg_0azi_ID1/IMG_KuSAR_H1H1_STR1_azbias1024.tif"
    xml_path = base / "Annotation/30deg_0azi_ID1/IMG_KuSAR_H1H1_STR1_azbias1024.xml"

    EvalApp(weights, Path("runs/pretrained_inference/eval")).run(image_path, xml_path)


if __name__ == "__main__":
    main()