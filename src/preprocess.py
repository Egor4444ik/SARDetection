from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import uniform_filter


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, sample): ...


class Despeckler(Preprocessor):
    def __init__(self, window=7, strength=1.0):
        self.window = window
        self.strength = strength

    def __call__(self, sample):
        magnitude = np.abs(sample)
        mean = uniform_filter(magnitude, self.window)
        variance = uniform_filter(magnitude ** 2, self.window) - mean ** 2
        noise = np.mean(variance)
        weight = variance / (variance + noise + 1e-8)
        filtered = mean + weight * (magnitude - mean)
        return self.strength * filtered + (1 - self.strength) * magnitude


class PauliDecomposer(Preprocessor):
    def __call__(self, sample):
        if sample.ndim != 3 or sample.shape[-1] < 3:
            return sample
        hh, hv, vv = sample[..., 0], sample[..., 1], sample[..., 2]
        odd = np.abs(hh + vv) / np.sqrt(2)
        double = np.abs(hh - vv) / np.sqrt(2)
        volume = np.abs(hv) * np.sqrt(2)
        return np.stack([odd, double, volume], axis=-1)


class VolumeSuppressor(Preprocessor):
    def __init__(self, factor=0.5):
        self.factor = factor

    def __call__(self, sample):
        if sample.shape[-1] < 3:
            return sample
        adjusted = sample.copy()
        adjusted[..., 2] *= self.factor
        return adjusted


class LogPercentileNormalizer(Preprocessor):
    def __init__(self, low=1, high=99):
        self.low = low
        self.high = high

    def __call__(self, sample):
        log = np.log1p(np.abs(sample))
        p_low, p_high = np.percentile(log, [self.low, self.high])
        clipped = np.clip(log, p_low, p_high)
        return (clipped - p_low) / (p_high - p_low + 1e-8)


class Pipeline(Preprocessor):
    def __init__(self, *stages):
        self.stages = stages

    def __call__(self, sample):
        result = sample
        for stage in self.stages:
            result = stage(result)
        return result


class PipelineBuilder:
    def __init__(self):
        self._stages = []

    def with_despeckling(self, method, **kwargs):
        if method and method != "none":
            self._stages.append(Despeckler(**kwargs))
        return self

    def with_pauli(self, enabled):
        if enabled:
            self._stages.append(PauliDecomposer())
        return self

    def with_volume_suppression(self, factor):
        self._stages.append(VolumeSuppressor(factor))
        return self

    def with_normalization(self, low, high):
        self._stages.append(LogPercentileNormalizer(low, high))
        return self

    def build(self):
        return Pipeline(*self._stages)