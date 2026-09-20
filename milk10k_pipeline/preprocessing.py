"""
preprocessing.py
Parte 3: funciones reutilizables de preprocesado.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def preprocess_image(source, size=(224, 224), color_space="rgb", normalization="minmax", stats=None):
    """
    Carga y preprocesa una imagen.

    Parametros
    ----------
    source : str | Path | np.ndarray
    size : (h, w)
    color_space : 'rgb' | 'gray'
    normalization : 'minmax' | 'zscore' | None
    stats : dict con 'mean' y 'std' (solo para zscore)

    Devuelve
    --------
    arr : np.ndarray
    info : dict con shape, dtype, min, max, normalization, color_space
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB").resize(size)
        arr = np.array(img)
    else:
        arr = np.asarray(source)
        if arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(np.uint8)).resize(size)
            arr = np.array(img)

    if color_space == "gray":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif color_space == "rgb":
        pass
    else:
        raise ValueError("color_space debe ser 'rgb' o 'gray'")

    arr = arr.astype(np.float32) / 255.0

    if normalization == "minmax":
        pass
    elif normalization == "zscore":
        if stats is None:
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std = arr.std(axis=(0, 1), keepdims=True) + 1e-8
        else:
            mean = np.array(stats["mean"], dtype=np.float32)
            std = np.array(stats["std"], dtype=np.float32)
            if arr.ndim == 3:
                mean = mean.reshape(1, 1, -1)
                std = std.reshape(1, 1, -1)
        arr = (arr - mean) / std
    elif normalization is None:
        pass
    else:
        raise ValueError("normalization debe ser 'minmax', 'zscore' o None")

    info = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "normalization": normalization,
        "color_space": color_space,
    }
    return arr, info


def preprocess_batch(sources, size=(224, 224), color_space="rgb", normalization="minmax", stats=None):
    """
    Aplica preprocess_image a una lista de paths o arrays.

    Devuelve
    --------
    batch : np.ndarray apilado (N, H, W, C) o (N, H, W)
    skipped : list de (source, error_msg)
    """
    processed = []
    skipped = []
    for src in sources:
        try:
            arr, _ = preprocess_image(src, size=size, color_space=color_space,
                                      normalization=normalization, stats=stats)
            processed.append(arr)
        except Exception as e:
            skipped.append((str(src), str(e)))
    if not processed:
        return np.array([]), skipped
    try:
        batch = np.stack(processed, axis=0)
    except ValueError:
        batch = processed
    return batch, skipped
