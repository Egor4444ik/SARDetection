from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# src/config.py
class Settings(BaseSettings):
    HUGGING_FACE_TOKEN: str

    REPO_ID: str = "waterdisappear/ATRNet-STAR"
    REPO_TYPE: str = "dataset"
    BASE_REPO_PATH: str = "Raw_data/Subset_City"

    @property
    def MANIFEST_REPO_PATH(self) -> str:
        return f"{self.BASE_REPO_PATH}/manifest.csv"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DatasetConfig(ABC):
    REQUIRED_KEYS = ("path", "train", "val", "nc", "names")
    SPLITS = ("train", "val", "test")

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._require_keys()
        self.mode = self._resolve_mode()
        self.channels = self._resolve_channels()
        self.num_classes = int(self.raw["nc"])
        self.names = self.raw["names"]
        self.image_size = self.raw.get("imgsz", 640)
        self.batch_size = self.raw.get("batch", 16)
        self.workers = self.raw.get("workers", 8)

    def _require_keys(self):
        missing = [key for key in self.REQUIRED_KEYS if key not in self.raw]
        if missing:
            raise ValueError(f"{self.config_path} missing: {missing}")

    def count_split(self, split):
        entry = self.raw.get(split)
        if entry is None:
            return 0
        base = Path(self.raw["path"])
        return self._count_list(base, entry) if entry.endswith(".txt") else self._count_directory(base, entry)

    def _count_list(self, base, entry):
        path = base / entry
        if not path.exists():
            path = self.config_path.parent / entry
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    def _count_directory(self, base, entry):
        path = base / entry
        return len(list(path.iterdir())) if path.is_dir() else 0

    def ultralytics_args(self, overrides):
        args = {
            "data": str(self.config_path),
            "epochs": overrides.epochs,
            "imgsz": overrides.imgsz or self.image_size,
            "batch": overrides.batch or self.batch_size,
            "workers": overrides.workers or self.workers,
            "device": overrides.device,
            "project": overrides.project,
            "name": overrides.name or self._default_run_name(),
            "patience": overrides.patience,
            "resume": overrides.resume,
            "pretrained": True,
            "exist_ok": True,
            "verbose": True,
            "save": True,
            "save_period": 10,
            "plots": True,
            "val": True,
        }
        if self.channels != 3:
            args["ch"] = self.channels
        return args

    def _default_run_name(self):
        return f"{self.mode}_{datetime.now():%Y%m%d_%H%M%S}"

    @abstractmethod
    def _resolve_mode(self): ...

    @abstractmethod
    def _resolve_channels(self): ...


class ImageConfig(DatasetConfig):
    def _resolve_mode(self):
        return "image"

    def _resolve_channels(self):
        return 3


class SARConfig(DatasetConfig):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.sar_options = self.raw.get("sar", {})

    def _resolve_mode(self):
        return "sar"

    def _resolve_channels(self):
        return int(self.raw.get("ch", 2))


class ConfigFactory:
    _MODES = {"image": ImageConfig, "sar": SARConfig}

    @classmethod
    def create(cls, config_path):
        return cls._MODES[cls._detect_mode(config_path)](config_path)

    @staticmethod
    def _detect_mode(config_path):
        name = Path(config_path).stem.lower()
        if "image" in name:
            return "image"
        if "sar" in name:
            return "sar"
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return "sar" if "sar" in raw else "image"

settings = Settings()