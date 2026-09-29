from abc import ABC, abstractmethod
from pathlib import Path
from ultralytics import YOLO


class ModelLoader:
    def __init__(self, weights):
        self.weights = weights

    def load(self):
        return YOLO(self.weights)


class Trainer(ABC):
    def __init__(self, config, model, overrides):
        self.config = config
        self.model = model
        self.overrides = overrides

    @abstractmethod
    def train(self): ...

    def _base_args(self):
        return self.config.ultralytics_args(self.overrides)


class ImageTrainer(Trainer):
    def train(self):
        return self.model.train(**self._base_args())


class SARTrainer(Trainer):
    def __init__(self, config, model, overrides, dataset_factory):
        super().__init__(config, model, overrides)
        self.dataset_factory = dataset_factory

    def train(self):
        args = self._base_args()
        args["_sar_config"] = self.config.sar_options
        args["trainer"] = self._build_trainer_class()
        return self.model.train(**args)

    def _build_trainer_class(self):
        from ultralytics.models.yolo.segment import SegmentationTrainer
        factory = self.dataset_factory
        components = self.config.sar_options.get("channels", ["I", "Q"])

        class SARSegmentationTrainer(SegmentationTrainer):
            def build_dataset(inner_self, img_path, mode="train", batch=None):
                paths = inner_self._read_paths(img_path)
                return factory.create_sar(paths, components)

            def get_dataloader(inner_self, dataset_path, batch_size=16, rank=0, mode="train"):
                return super().get_dataloader(dataset_path, batch_size, rank, mode)

            @staticmethod
            def _read_paths(source):
                path = Path(source)
                if path.is_file():
                    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
                return [str(p) for p in path.iterdir()]

        return SARSegmentationTrainer


class TrainerFactory:
    _TRAINERS = {"image": ImageTrainer, "sar": SARTrainer}

    @classmethod
    def create(cls, config, model, overrides, dataset_factory=None):
        trainer_cls = cls._TRAINERS[config.mode]
        return trainer_cls(config, model, overrides, dataset_factory) \
            if trainer_cls is SARTrainer \
            else trainer_cls(config, model, overrides)