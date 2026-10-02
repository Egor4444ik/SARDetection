from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from src.config import settings


@dataclass
class SceneFiles:
    folder: str
    image_path: Path
    xml_path: Path


@dataclass
class TilePrediction:
    tile: Path
    cls: int
    conf: float


class SarPreprocessor:
    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        self._low = low
        self._high = high

    def to_rgb8(self, image: Image.Image) -> Image.Image:
        arr = np.array(image).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, [self._low, self._high])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        uint8 = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(np.stack([uint8] * 3, axis=-1), mode="RGB")


class SceneFetcher:
    REPO_ID = "waterdisappear/ATRNet-STAR"
    REPO_TYPE = "dataset"
    BASE = "Raw_data/Subset_Sandstone"

    def __init__(self, local_dir: Path) -> None:
        self._local_dir = local_dir
        self._token = settings.HUGGING_FACE_TOKEN

    def fetch(self, folder: str, image_name: str, xml_name: str) -> SceneFiles:
        image_path = Path(hf_hub_download(
            repo_id=self.REPO_ID,
            filename=f"{self.BASE}/Result/{folder}/{image_name}",
            repo_type=self.REPO_TYPE,
            local_dir=str(self._local_dir),
            token=self._token,
        ))
        xml_path = Path(hf_hub_download(
            repo_id=self.REPO_ID,
            filename=f"{self.BASE}/Annotation/{folder}/{xml_name}",
            repo_type=self.REPO_TYPE,
            local_dir=str(self._local_dir),
            token=self._token,
        ))
        return SceneFiles(folder, image_path, xml_path)


class ImageTileSplitter:
    def __init__(
        self,
        tile: int = 128,
        stride: int | None = None,
        preprocessor: SarPreprocessor | None = None,
    ) -> None:
        self._tile = tile
        self._stride = stride or tile
        self._pre = preprocessor or SarPreprocessor()

    def split(self, image_path: Path, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles: list[Path] = []
        with Image.open(image_path) as img:
            rgb = self._pre.to_rgb8(img)
            w, h = rgb.size
            idx = 0
            for y in range(0, h - self._tile + 1, self._stride):
                for x in range(0, w - self._tile + 1, self._stride):
                    crop = rgb.crop((x, y, x + self._tile, y + self._tile))
                    tile_path = out_dir / f"tile_{idx:05d}.png"
                    crop.save(tile_path)
                    tiles.append(tile_path)
                    idx += 1
        return tiles


class Detector:
    def __init__(self, weights: Path, conf: float = 0.01) -> None:
        self._model = YOLO(str(weights))
        self._conf = conf

    @property
    def names(self) -> dict[int, str]:
        return dict(self._model.names)

    def predict_many(self, tiles: list[Path], imgsz: int = 128) -> list[TilePrediction]:
        results = self._model.predict(
            source=[str(t) for t in tiles],
            imgsz=imgsz,
            conf=self._conf,
            verbose=False,
            stream=True,
        )
        out: list[TilePrediction] = []
        for tile, r in zip(tiles, results):
            for box in r.boxes:
                out.append(TilePrediction(
                    tile=tile,
                    cls=int(box.cls.item()),
                    conf=float(box.conf.item()),
                ))
        return out


class Reporter:
    def __init__(self, names: dict[int, str]) -> None:
        self._names = names
        self._per_class: dict[int, int] = {}
        self._total = 0

    def update(self, preds: list[TilePrediction]) -> None:
        for p in preds:
            self._total += 1
            self._per_class[p.cls] = self._per_class.get(p.cls, 0) + 1
            print(f"  {p.tile.name}: {self._names.get(p.cls, p.cls)} {p.conf:.3f}")

    def report(self) -> None:
        print(f"\ntotal detections: {self._total}")
        if not self._per_class:
            return
        print("per class:")
        for cls, count in sorted(self._per_class.items(), key=lambda x: -x[1]):
            print(f"  {self._names.get(cls, cls)}: {count}")


class InferenceApp:
    def __init__(self, weights: Path, work_dir: Path, tile: int = 128) -> None:
        self._work_dir = work_dir
        self._fetcher = SceneFetcher(work_dir / "scene")
        self._splitter = ImageTileSplitter(tile)
        self._detector = Detector(weights)

    def run(self, folder: str, image_name: str, xml_name: str) -> None:
        scene = self._fetcher.fetch(folder, image_name, xml_name)
        print(f"scene: {scene.image_path}")

        tiles_dir = self._work_dir / "tiles" / folder
        tiles = self._splitter.split(scene.image_path, tiles_dir)
        print(f"tiles: {len(tiles)}")

        preds = self._detector.predict_many(tiles)
        reporter = Reporter(self._detector.names)
        reporter.update(preds)
        reporter.report()


def main() -> None:
    weights = Path(
        "data/_bench/ATRBench/Detection/unpacked/weight/"
        "v8/SOC_40classes/SOC_40classes_train/weights/best.pt"
    )
    work_dir = Path("runs/pretrained_inference")
    app = InferenceApp(weights, work_dir, tile=128)
    app.run(
        folder="30deg_0azi_ID1",
        image_name="IMG_KuSAR_H1H1_STR1_azbias1024.tif",
        xml_name="IMG_KuSAR_H1H1_STR1_azbias1024.xml",
    )


if __name__ == "__main__":
    main()