import typing as T

import numpy as np

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

from operation import pool
from wavelet import ComplexMorletBank


class ScatteringNetwork:
    

    def __init__(
        self,
        *layer_kwargs: dict,
        bins: int = 128,
        sampling_rate: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.bins = bins
        self.verbose = verbose
        self.banks = [
            ComplexMorletBank(bins, sampling_rate=sampling_rate, **kw)
            for kw in layer_kwargs
        ]

    def __len__(self) -> int:
        return len(self.banks)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"bins={self.bins}, "
            f"sampling_rate={self.sampling_rate}, "
            f"len={len(self)})"
            "\n"
        ) + "\n".join(str(bank) for bank in self.banks)

    def transform_segment(
        self,
        segment: np.ndarray,
        reduce_type: T.Union[T.Callable, None] = None,
    ) -> list:
        
        output = list()
        for bank in self.banks:
            scalogram = bank.transform(segment)
            segment = scalogram
            output.append(pool(scalogram, reduce_type))

        return output

    def transform(
        self,
        segments: np.ndarray,
        reduce_type: T.Union[T.Callable, None] = None,
    ) -> list:
        

        features = [[] for _ in range(len(self))]
        iter = tqdm(segments) if self.verbose else segments
        for segment in iter:
            scatterings = self.transform_segment(segment, reduce_type)
            for layer_index, scattering in enumerate(scatterings):
                features[layer_index].append(scattering)

        return [np.array(feature) for feature in features]
