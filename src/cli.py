import argparse
from src.config import ConfigFactory
from src.readers import default_registry
from src.preprocess import PipelineBuilder
from src.data import DatasetFactory
from src.model import ModelLoader, TrainerFactory


class Arguments:
    def __init__(self, namespace):
        self.config = namespace.config
        self.model = namespace.model
        self.epochs = namespace.epochs
        self.project = namespace.project
        self.name = namespace.name
        self.device = namespace.device
        self.resume = namespace.resume
        self.patience = namespace.patience
        self.workers = namespace.workers
        self.batch = namespace.batch
        self.imgsz = namespace.imgsz


class Parser:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--model", default="yolo26n-seg.pt")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--project", default="runs/atrnet_star")
        parser.add_argument("--name", default=None)
        parser.add_argument("--device", default="0")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--patience", type=int, default=20)
        parser.add_argument("--workers", type=int, default=None)
        parser.add_argument("--batch", type=int, default=None)
        parser.add_argument("--imgsz", type=int, default=None)
        return Arguments(parser.parse_args())


class Application:
    def __init__(self):
        self.registry = default_registry()
        self.builder_cls = PipelineBuilder
        self.dataset_factory_cls = DatasetFactory
        self.model_loader_cls = ModelLoader
        self.trainer_factory = TrainerFactory
        self.config_factory = ConfigFactory

    def run(self, arguments):
        config = self.config_factory.create(arguments.config)
        self.report(config)
        model = self.model_loader_cls(arguments.model).load()
        dataset_factory = self.dataset_factory_cls(self.registry, self.build_pipeline(config))
        return self.trainer_factory.create(config, model, arguments, dataset_factory).train()

    def build_pipeline(self, config):
        builder = self.builder_cls()
        if config.mode == "sar":
            options = config.sar_options
            builder = (builder
                       .with_despeckling(options.get("despeckle", "none"))
                       .with_pauli(options.get("pauli", False))
                       .with_volume_suppression(options.get("volume_factor", 0.5))
                       .with_normalization(
                           options.get("percentile_low", 1),
                           options.get("percentile_high", 99)))
        else:
            builder = builder.with_normalization(1, 99)
        return builder.build()

    def report(self, config):
        print(f"mode={config.mode} classes={config.num_classes} channels={config.channels}")
        for split in config.SPLITS:
            count = config.count_split(split)
            if count:
                print(f"  {split}: {count}")