from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from src.config import settings


class Downloader:
    REPO_ID = "waterdisappear/ATRNet-STAR"
    REPO_TYPE = "dataset"

    def __init__(self, local_root: Path) -> None:
        self._local_root = local_root
        self._token = settings.HUGGING_FACE_TOKEN

    def download(self, files: list[str]) -> None:
        for filename in files:
            try:
                path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=filename,
                    repo_type=self.REPO_TYPE,
                    local_dir=str(self._local_root),
                    token=self._token,
                )
                print(f"OK   {filename} -> {path}")
            except Exception as e:
                print(f"ERR  {filename}: {type(e).__name__}: {e}")


def main() -> None:
    local_root = Path("./data/_downloads")
    local_root.mkdir(parents=True, exist_ok=True)

    downloader = Downloader(local_root)

    # 1. Веса YOLO (детекция)
    yolo_weights = [
        "ATRBench/Detection/yolo8_and_yolo10.7z",
    ]
    downloader.download(yolo_weights)

    # 2. Веса ResNet (классификация)
    resnet_weights = [
        "ATRBench/Classification/ResNet18.zip",
        "ATRBench/Classification/ResNet34.zip",
    ]
    downloader.download(resnet_weights)

    # 3. Веса ViT (классификация)
    vit_weights = [
        "ATRBench/Classification/ViT.zip",
    ]
    downloader.download(vit_weights)

    # 4. Датасет SOC-40 (основной протокол)
    soc40_data = [
        "Ground_Range/Amplitude_8bit/Amplitude 8-bit data_地距幅度8位数据.7z.001",
        "Ground_Range/Amplitude_8bit/Amplitude 8-bit data_地距幅度8位数据.7z.002",
        "Ground_Range/Amplitude_8bit/Amplitude 8-bit data_地距幅度8位数据.7z.003",
    ]
    downloader.download(soc40_data)

    # 5. Аннотации COCO для SOC-40 (если ещё не скачаны)
    soc40_annotations = [
        "Ground_Range/地距检测标注和转换代码_Annotation_COCO.zip.zip",
    ]
    downloader.download(soc40_annotations)


if __name__ == "__main__":
    main()