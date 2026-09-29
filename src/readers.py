from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import tifffile
from scipy.io import loadmat


class Reader(ABC):
    SUPPORTED_EXTENSIONS = ()

    @abstractmethod
    def read(self, path): ...

    @classmethod
    def supports(cls, path):
        return Path(path).suffix.lower() in cls.SUPPORTED_EXTENSIONS


class TIFReader(Reader):
    SUPPORTED_EXTENSIONS = (".tif", ".tiff")

    def read(self, path):
        return tifffile.imread(str(path))


class MATReader(Reader):
    SUPPORTED_EXTENSIONS = (".mat",)

    def read(self, path):
        return self._largest_array(loadmat(str(path)))

    @staticmethod
    def _largest_array(data):
        arrays = [v for k, v in data.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        return max(arrays, key=lambda arr: arr.size) if arrays else None


class SLCReader(Reader):
    SUPPORTED_EXTENSIONS = (".slc",)

    def read(self, path):
        raw = np.fromfile(str(path), dtype=np.complex64)
        side = int(np.sqrt(raw.size))
        return raw.reshape(side, side)


class ReaderRegistry:
    def __init__(self):
        self._readers = []

    def register(self, reader_cls):
        self._readers.append(reader_cls)
        return self

    def resolve(self, path):
        for reader_cls in self._readers:
            if reader_cls.supports(path):
                return reader_cls()
        raise ValueError(f"no reader for {path}")


def default_registry():
    return ReaderRegistry().register(TIFReader).register(MATReader).register(SLCReader)