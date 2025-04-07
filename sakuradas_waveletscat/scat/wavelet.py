import typing as T
try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp
import numpy as np
from scipy.signal.windows import tukey


def gaussian_window(
    x: xp.ndarray,
    width: T.Union[float, T.Sequence[float], xp.ndarray],
) -> xp.ndarray:
    
    x = xp.array(x)
    width = xp.array(width)
    width = width[:, None] if width.shape and (width.ndim == 1) else width

    return xp.exp(-((x / width) ** 2))


def complex_morlet(
    x: xp.ndarray,
    center: T.Union[float, T.Sequence[float], xp.ndarray],
    width: T.Union[float, T.Sequence[float], xp.ndarray],
) -> xp.ndarray:
    
    x = xp.array(x)
    width = xp.array(width)
    center = xp.array(center)

    width = width[:, None] if width.shape else width
    center = center[:, None] if center.shape else center

    if width.shape and center.shape:
        assert (
            width.shape == center.shape
        ), f"Shape for widths {width.shape} and centers {center.shape} differ."

    return gaussian_window(x, width) * xp.exp(2j * xp.pi * center * x)


class ComplexMorletBank:
    """Complex Morlet filter bank."""

    def __init__(
        self,
        bins: int,
        octaves: int = 8,
        resolution: int = 1,
        quality: float = 4.0,
        taper_alpha=None,
        sampling_rate: float = 1.0,
    ):
        
        self.bins = bins
        self.octaves = octaves
        self.resolution = resolution
        self.quality = quality
        self.sampling_rate = sampling_rate

        self.wavelets = complex_morlet(self.times, self.centers, self.widths)
        self.spectra = xp.fft.fft(self.wavelets)
        self.size = self.wavelets.shape[0]

        if taper_alpha is None:
            self.taper = xp.array(xp.ones(bins))
        else:
            self.taper = xp.array(tukey(bins, alpha=taper_alpha))

    def __repr__(self) -> str:
        return (
            f"ComplexMorletBank(bins={self.bins}, octaves={self.octaves}, "
            f"resolution={self.resolution}, quality={self.quality}, "
            f"sampling_rate={self.sampling_rate}, len={len(self)})"
        )

    def __len__(self) -> int:
        return self.octaves * self.resolution

    def transform(self, segment: xp.ndarray) -> np.ndarray:
        segment = xp.fft.fft(xp.array(segment) * xp.array(self.taper))
        convolved = segment[..., None, :] * xp.array(self.spectra)
        scalogram = xp.fft.fftshift(xp.fft.ifft(convolved), axes=-1)
        if xp.__name__ == "cupy":
            return np.abs(xp.asnumpy(scalogram))
        else:
            return xp.abs(scalogram)

    @property
    def times(self) -> np.ndarray:
        duration = self.bins / self.sampling_rate
        if xp.__name__ == "cupy":
            return xp.asnumpy(xp.linspace(-0.5, 0.5, num=self.bins) * duration)
        else:
            return xp.linspace(-0.5, 0.5, num=self.bins) * duration

    @property
    def frequencies(self) -> np.ndarray:
        if xp.__name__ == "cupy":
            return xp.asnumpy(xp.linspace(0, self.sampling_rate, self.bins))
        else:
            return xp.linspace(0, self.sampling_rate, self.bins)

    @property
    def nyquist(self) -> float:
        return self.sampling_rate / 2

    @property
    def shape(self) -> tuple:
        return len(self), self.bins

    @property
    def ratios(self) -> np.ndarray:
        ratios = xp.linspace(self.octaves, 0.0, self.shape[0], endpoint=False)
        if xp.__name__ == "cupy":
            return xp.asnumpy(-ratios[::-1])
        else:
            return -ratios[::-1]

    @property
    def scales(self) -> np.ndarray:
        if xp.__name__ == "cupy":
            return xp.asnumpy(2**self.ratios)
        else:
            return 2**self.ratios

    @property
    def centers(self) -> np.ndarray:
        if xp.__name__ == "cupy":
            return xp.asnumpy(self.scales * self.nyquist)
        else:
            return self.scales * self.nyquist

    @property
    def widths(self) -> np.ndarray:
        if xp.__name__ == "cupy":
            return xp.asnumpy(self.quality / self.centers)
        else:
            return self.quality / self.centers
