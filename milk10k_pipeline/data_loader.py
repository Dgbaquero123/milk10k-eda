"""
data_loader.py
Part 4: batched data loader for training.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from .preprocessing import preprocess_image

TARGET = "diagnosis_1"


class Milk10kLoader:
    """
    Loader que produce batches (X, y) a partir del metadata y el directorio
    de imagenes.

    Parametros
    ----------
    metadata : pd.DataFrame
    images_dir : str | Path
    batch_size : int
    size : (h, w)
    color_space : 'rgb' | 'gray'
    normalization : 'minmax' | 'zscore' | None
    shuffle : bool
    seed : int

    Iterar sobre el loader produce tuplas (X, y):
        X : np.ndarray (B, H, W, C) o (B, H, W)
        y : np.ndarray de strings (etiquetas)
    """

    def __init__(self, metadata, images_dir, batch_size=16, size=(224, 224),
                 color_space="rgb", normalization="minmax",
                 shuffle=True, seed=0):
        self.images_dir = Path(images_dir)
        self.batch_size = batch_size
        self.size = size
        self.color_space = color_space
        self.normalization = normalization
        self.shuffle = shuffle
        self.seed = seed

        df = metadata.copy()
        df["path"] = df["isic_id"].apply(lambda x: self.images_dir / f"{x}.jpg")
        df = df[df["path"].apply(lambda p: p.exists())]
        df = df[df[TARGET].notna()]
        self.df = df.reset_index(drop=True)

        self._classes = sorted(self.df[TARGET].unique().tolist())
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    @property
    def classes(self):
        return self._classes

    def _iter_batches(self):
        idx = np.arange(len(self.df))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            yield idx[start:start + self.batch_size]

    def __iter__(self):
        for batch_idx in self._iter_batches():
            rows = self.df.iloc[batch_idx]
            X_list = []
            y_list = []
            for _, row in rows.iterrows():
                try:
                    arr, _ = preprocess_image(row["path"], size=self.size,
                                              color_space=self.color_space,
                                              normalization=self.normalization)
                    X_list.append(arr)
                    y_list.append(row[TARGET])
                except Exception:
                    continue
            if not X_list:
                continue
            X = np.stack(X_list, axis=0)
            y = np.array(y_list)
            yield X, y
