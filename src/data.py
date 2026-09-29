from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset


class ChannelExtractor:
    def __init__(self, components):
        self.components = components

    def __call__(self, complex_data):
        available = {
            "I": np.real(complex_data),
            "Q": np.imag(complex_data),
            "amplitude": np.abs(complex_data),
            "phase": np.angle(complex_data),
        }
        return np.stack([available[c] for c in self.components], axis=-1)


class BaseDataset(Dataset, ABC):
    def __init__(self, paths, preprocessor):
        self.paths = list(paths)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.paths)

    @abstractmethod
    def __getitem__(self, index): ...


class ImageDataset(BaseDataset):
    def __init__(self, paths, preprocessor, reader):
        super().__init__(paths, preprocessor)
        self.reader = reader

    def __getitem__(self, index):
        image = self.reader.read(self.paths[index])
        processed = self.preprocessor(image)
        tensor = self._to_tensor(processed)
        return tensor, str(self.paths[index])

    @staticmethod
    def _to_tensor(image):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)


class SARDataset(BaseDataset):
    def __init__(self, paths, preprocessor, reader, extractor):
        super().__init__(paths, preprocessor)
        self.reader = reader
        self.extractor = extractor

    def __getitem__(self, index):
        complex_data = self.reader.read(self.paths[index])
        channels = self.extractor(complex_data)
        processed = self.preprocessor(channels)
        return self._to_tensor(processed), str(self.paths[index])

    @staticmethod
    def _to_tensor(array):
        tensor = torch.from_numpy(array.astype(np.float32))
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor.permute(2, 0, 1)


class DatasetFactory:
    def __init__(self, registry, preprocessor):
        self.registry = registry
        self.preprocessor = preprocessor

    def create_image(self, paths):
        reader = self.registry.resolve(paths[0])
        return ImageDataset(paths, self.preprocessor, reader)

    def create_sar(self, paths, components):
        reader = self.registry.resolve(paths[0])
        extractor = ChannelExtractor(components)
        return SARDataset(paths, self.preprocessor, reader, extractor)