import typing as T
import numpy as np

def segment(
    x: np.ndarray,
    window_size: int,
    stride: T.Union[int, None] = None,
) -> T.Generator[np.ndarray, None, None]:
    
    bins = x.shape[-1]
    index = 0
    stride = window_size if stride is None else stride
    while (index + window_size) <= bins:
        yield x[..., index : index + window_size]
        index += stride


def segmentize(
    x: np.ndarray,
    window_size: int,
    stride: T.Union[int, None] = None,
) -> np.ndarray:
    
    return np.array([x for x in segment(x, window_size, stride)])


def pool(
    x: np.ndarray,
    reduce_type: T.Union[T.Callable, None] = None,
) -> np.ndarray:
    
    if reduce_type is None:
        return x
    else:
        return reduce_type(x, axis=-1)
