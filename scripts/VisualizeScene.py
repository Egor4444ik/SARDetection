from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


@dataclass
class GtObject:
    type_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Detection:
    cls: int
    conf: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class SarImageLoader:
    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        self._low = low
        self._high = high

    def load_rgb(self, path: Path) -> Image.Image:
        arr = np.array(Image.open(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = np.percentile(arr, [self._low, self._high])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() + 1e-6)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        uint8 = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(np.stack([uint8] * 3, axis=-1), mode="RGB")


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


class TiledDetector:
    def __init__(self, weights: Path, tile: int = 128, conf: float = 0.05) -> None:
        self._model = YOLO(str(weights))
        self._tile = tile
        self._conf = conf

    @property
    def names(self) -> dict[int, str]:
        return dict(self._model.names)

    def detect(self, image: Image.Image) -> list[Detection]:
        w, h = image.size
        arr = np.array(image)

        tiles: list[np.ndarray] = []
        origins: list[tuple[int, int]] = []
        for y in range(0, h - self._tile + 1, self._tile):
            for x in range(0, w - self._tile + 1, self._tile):
                tiles.append(arr[y:y + self._tile, x:x + self._tile])
                origins.append((x, y))

        detections: list[Detection] = []
        results = self._model.predict(
            source=tiles, imgsz=self._tile, conf=self._conf, verbose=False, stream=True
        )
        for (ox, oy), r in zip(origins, results):
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                detections.append(Detection(
                    cls=int(box.cls.item()),
                    conf=float(box.conf.item()),
                    xmin=xyxy[0] + ox,
                    ymin=xyxy[1] + oy,
                    xmax=xyxy[2] + ox,
                    ymax=xyxy[3] + oy,
                ))
        return detections


class GtScaler:
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def apply(self, gts: list[GtObject]) -> list[GtObject]:
        return [
            GtObject(
                type_name=g.type_name,
                xmin=int(g.xmin * self._scale),
                ymin=int(g.ymin * self._scale),
                xmax=int(g.xmax * self._scale),
                ymax=int(g.ymax * self._scale),
            )
            for g in gts
        ]


class DrawingStyles:
    def __init__(self, font_size: int = 16) -> None:
        self._gt_color = (0, 255, 0)
        self._pred_color = (255, 64, 64)
        self._font_size = font_size

    @property
    def gt_color(self) -> tuple[int, int, int]:
        return self._gt_color

    @property
    def pred_color(self) -> tuple[int, int, int]:
        return self._pred_color

    def font(self) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("Arial.ttf", self._font_size)
        except OSError:
            return ImageFont.load_default()


class SceneRenderer:
    def __init__(self, styles: DrawingStyles) -> None:
        self._styles = styles
        self._font = styles.font()

    def render_gt(
        self, image: Image.Image, gts: list[GtObject]
    ) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for g in gts:
            draw.rectangle(
                [g.xmin, g.ymin, g.xmax, g.ymax],
                outline=self._styles.gt_color,
                width=3,
            )
            self._label(draw, (g.xmin, g.ymin), f"GT: {g.type_name}", self._styles.gt_color)
        return canvas

    def render_pred(
        self, image: Image.Image, dets: list[Detection], names: dict[int, str]
    ) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for d in dets:
            draw.rectangle(
                [d.xmin, d.ymin, d.xmax, d.ymax],
                outline=self._styles.pred_color,
                width=2,
            )
            self._label(
                draw,
                (d.xmin, d.ymax),
                f"{names.get(d.cls, d.cls)} {d.conf:.2f}",
                self._styles.pred_color,
            )
        return canvas

    def render_overlay(
        self,
        image: Image.Image,
        gts: list[GtObject],
        dets: list[Detection],
        names: dict[int, str],
    ) -> Image.Image:
        canvas = self.render_gt(image, gts)
        draw = ImageDraw.Draw(canvas)
        for d in dets:
            draw.rectangle(
                [d.xmin, d.ymin, d.xmax, d.ymax],
                outline=self._styles.pred_color,
                width=2,
            )
            self._label(
                draw,
                (d.xmin, d.ymax),
                f"{names.get(d.cls, d.cls)} {d.conf:.2f}",
                self._styles.pred_color,
            )
        return canvas

    def _label(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        x, y = xy
        bbox = draw.textbbox((x, y), text, font=self._font)
        pad = 3
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(0, 0, 0),
        )
        draw.text((x, y), text, fill=color, font=self._font)


class SceneVisualizer:
    def __init__(
        self,
        weights: Path,
        out_dir: Path,
        conf: float = 0.05,
        max_side: int = 2048,
        tile: int = 128,
    ) -> None:
        self._loader = SarImageLoader()
        self._detector = TiledDetector(weights, tile=tile, conf=conf)
        self._renderer = SceneRenderer(DrawingStyles())
        self._out_dir = out_dir
        self._max_side = max_side

    def run(self, image_path: Path, xml_path: Path, tag: str) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)

        raw = self._loader.load_rgb(image_path)
        w0, h0 = raw.size
        scaled = self._downscale(raw)
        w1, h1 = scaled.size
        scale = w1 / w0

        gts_raw = XmlReader(xml_path).read()
        gts = GtScaler(scale).apply(gts_raw)
        dets = self._detector.detect(scaled)

        print(f"  scale={scale:.3f} gt={len(gts)} pred={len(dets)}")

        self._save(scaled, f"{tag}_raw.png")
        self._save(self._renderer.render_gt(scaled, gts), f"{tag}_gt.png")
        self._save(
            self._renderer.render_pred(scaled, dets, self._detector.names),
            f"{tag}_pred.png",
        )
        self._save(
            self._renderer.render_overlay(scaled, gts, dets, self._detector.names),
            f"{tag}_overlay.png",
        )

    def _downscale(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self._max_side / max(w, h)
        if scale >= 1.0:
            return image
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    def _save(self, image: Image.Image, name: str) -> None:
        path = self._out_dir / name
        image.save(path)
        print(f"  saved: {path}")


def main() -> None:
    import json
    from pathlib import Path

    weights = Path(
        "data/_bench/ATRBench/Detection/unpacked/weight/"
        "v8/SOC_40classes/SOC_40classes_train/weights/best.pt"
    )
    base = Path("runs/pretrained_inference/scene/Raw_data/Subset_Sandstone")
    index_path = Path("runs/pretrained_inference/sandstone_index.json")

    if not index_path.exists():
        print(f"нет индекса: {index_path}")
        return

    index = json.loads(index_path.read_text())
    if not index:
        print("индекс пустой")
        return

    viz = SceneVisualizer(
        weights,
        Path("runs/pretrained_inference/visuals"),
        conf=0.05,
        max_side=2048,
    )

    for folder, names in sorted(index.items()):
        tif = names.get("tif")
        xml = names.get("xml")
        if not tif or not xml:
            print(f"skip {folder}: нет имён")
            continue

        img = base / "Result" / folder / tif
        xml_p = base / "Annotation" / folder / xml
        if not img.exists() or not xml_p.exists():
            print(f"skip {folder}: файлов нет локально ({img.name})")
            continue

        print(f"render {folder}")
        viz.run(img, xml_p, tag=folder)


if __name__ == "__main__":
    main()