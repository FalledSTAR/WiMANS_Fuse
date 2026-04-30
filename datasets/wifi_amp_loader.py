from pathlib import Path

import numpy as np
import torch


def pad_or_truncate_time(array: np.ndarray, target_len: int, pad_mode: str = "left", truncate_mode: str = "tail") -> np.ndarray:
    if array.ndim != 4 or array.shape[1:] != (3, 3, 30):
        raise ValueError(f"Expected CSI amplitude shape [T,3,3,30], got {array.shape}")

    length = array.shape[0]
    if length == target_len:
        return array

    if length > target_len:
        if truncate_mode == "tail":
            return array[-target_len:]
        if truncate_mode == "head":
            return array[:target_len]
        raise ValueError(f"Unsupported truncate_mode: {truncate_mode}")

    pad_len = target_len - length
    pad_width = ((pad_len, 0), (0, 0), (0, 0), (0, 0)) if pad_mode == "left" else ((0, pad_len), (0, 0), (0, 0), (0, 0))
    return np.pad(array, pad_width, mode="constant")


def normalize_csi(array: np.ndarray, mode: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if mode in (None, "none"):
        return array

    if mode == "log1p_zscore":
        array = np.log1p(np.maximum(array, 0.0))
        mean = float(array.mean())
        std = float(array.std())
        return (array - mean) / max(std, 1e-6)

    if mode == "zscore":
        mean = float(array.mean())
        std = float(array.std())
        return (array - mean) / max(std, 1e-6)

    raise ValueError(f"Unsupported CSI normalization mode: {mode}")


def load_wifi_amplitude(path: str, target_len: int = 3000, pad_mode: str = "left", truncate_mode: str = "tail", normalize: str = "log1p_zscore") -> torch.Tensor:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)

    array = np.load(path_obj)
    array = pad_or_truncate_time(array, target_len=target_len, pad_mode=pad_mode, truncate_mode=truncate_mode)
    array = normalize_csi(array, mode=normalize)

    array = array.reshape(target_len, -1).T
    return torch.from_numpy(array.astype(np.float32, copy=False))
