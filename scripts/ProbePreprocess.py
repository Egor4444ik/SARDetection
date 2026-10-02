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


class Preprocessor:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def to_rgb(self, arr: np.ndarray) -> Image.Image:
        if self._mode == "raw":
            u = (arr / max(arr.max(), 1e-6) * 255).astype(np.uint8)
        elif self._mode == "pct":
            lo, hi = np.percentile(arr, [1, 99])
            u = (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        elif self._mode == "log":
            b = np.log1p(arr)
            lo, hi = np.percentile(b, [1, 99])
            u = (np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        else:
            raise ValueError(self._mode)
        return Image.fromarray(np.stack([u] * 3, -1), mode="RGB")


class ChipCutter:
    def __init__(self, size: int = 128) -> None:
        self._size = size

    def cut(self, image: Image.Image, obj: GtObject) -> Image.Image:
        cx = (obj.xmin + obj.xmax) // 2
        cy = (obj.ymin + obj.ymax) // 2
        half = self._size // 2
        left = max(0, min(image.width - self._size, cx - half))
        top = max(0, min(image.height - self._size, cy - half))
        return image.crop((left, top, left + self._size, top + self._size))


class ProbeRunner:
    def __init__(self, weights: Path, image_path: Path, xml_path: Path) -> None:
        self._model = YOLO(str(weights))
        self._image = np.array(Image.open(image_path)).astype(np.float32)
        self._objects = XmlReader(xml_path).read()[:10]
        self._cutter = ChipCutter(128)

    def run(self) -> None:
        for mode in ("raw", "pct", "log"):
            prep = Preprocessor(mode)
            rgb = prep.to_rgb(self._image)
            print(f"\n=== {mode} ===")
            for i, obj in enumerate(self._objects):
                chip = self._cutter.cut(rgb, obj)
                r = self._model.predict(source=np.array(chip), imgsz=128, verbose=False)
                preds = [
                    (self._model.names[int(b.cls.item())], float(b.conf.item()))
                    for b in r[0].boxes
                ]
                print(f"  [{i:02d}] gt={obj.type_name:30s} pred={preds}")


def main() -> None:
    weights = Path(
        "data/_bench/ATRBench/Detection/unpacked/weight/"
        "v8/SOC_40classes/SOC_40classes_train/weights/best.pt"
    )
    base = Path("runs/pretrained_inference/scene/Raw_data/Subset_Sandstone")
    folder = "30deg_0azi_ID1"
    image = base / "Result" / folder / "IMG_KuSAR_V1V1_STR1_azbias1024.tif"
    xml = base / "Annotation" / folder / "IMG_KuSAR_V1V1_STR1_azbias1024.xml"
    ProbeRunner(weights, image, xml).run()


if __name__ == "__main__":
    main()