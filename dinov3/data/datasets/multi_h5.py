import os
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Optional, Tuple
from torchvision.datasets import VisionDataset


class MultiH5Dataset(VisionDataset):
    """Dataset for loading images and annotations from multiple H5 files.

    Supports both discrete and continuous targets. For continuous values,
    annotations are discretized into bins using np.histogram().
    """

    NORMALIZE_MEAN = (0.5475857690770103,)
    NORMALIZE_STD = (0.12047245612891117,)

    def __init__(
        self,
        root: str,
        *,
        h5_key: str = "images",
        h5_annotations_key: str = "annotations",
        target_key: str = "",
        target_key_bins: int = 20,
        h5_arr_list_path: str = "",
        parquet_annotations: str = '',
        transform=None,
        target_transform=None,
        transforms=None,
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.transforms = transforms
        self.h5_key = h5_key
        self.h5_annotations_key = h5_annotations_key
        self.target_key = target_key
        self.target_key_bins = target_key_bins
        self.h5_arr_list_path = h5_arr_list_path
        self.parquet_annotations = parquet_annotations
        self.annotations = []

        # Discover .h5 files
        self.h5_paths = self._discover_h5_paths(root)
        if not self.h5_paths:
            raise RuntimeError(f"No .h5 files found in directory: {root}")

        # Optional filtering
        if len(self.h5_arr_list_path):
            h5_list = set(np.load(self.h5_arr_list_path, allow_pickle=True).tolist())
            self.h5_paths = [p for p in self.h5_paths if os.path.basename(p) in h5_list]

        # Open handles lazily
        self._handles: List[Optional[h5py.File]] = [None] * len(self.h5_paths)
        if len(self.parquet_annotations):
            #Set target key
            self.target_key = self.parquet_annotations.split('/')[-1].split('.')[0]
            #Load Parquet file
            self.df_selected = pd.read_parquet(parquet_annotations)
            #Pick index for the files and particles
            self._index = np.array((self.df_selected[['FileIndex','FileParticleIndex']])).astype(int)
            #And their respective annotations
            self.anns = self.df_selected['AnnotationValue']
            self.ann_bin = self.df_selected['BinCenter']
            self.ann_idx = self.df_selected['BinIndex'] #np.unique(self.df_selected['BinIndex'],return_inverse=True)[1]
        else:
            self._index, self.annotations = self._build_index()

        # if len(self.target_key):
        #     if self.target_key in ("Symmetry", "ProteinClass","ClassNumber"):
        #         # Discrete classes
        #         self.class_names, self.ann_idx = np.unique(self.annotations, return_inverse=True)
        #     else:
        #         # Continuous classes (binned)
        #         hist, bin_edges = np.histogram(self.annotations, bins=self.target_key_bins)
        #         self.ann_idx = np.digitize(self.annotations, bin_edges, right=True) - 1
        #         self.ann_idx = np.clip(self.ann_idx, 0, self.target_key_bins - 1)
        #         # Bin centers used as labels
        #         self.class_names = (bin_edges[:-1] + bin_edges[1:]) / 2
        #
        #     if len(self._index) != len(self.annotations):
        #         raise ValueError("Global annotation count mismatch between images and annotations")

        self.normalize_mean = self.NORMALIZE_MEAN
        self.normalize_std = self.NORMALIZE_STD

    def _discover_h5_paths(self, root: str) -> List[str]:
        return sorted(
            os.path.join(root, entry)
            for entry in os.listdir(root)
            if entry.endswith(".h5")
        )

    def _build_index(self) -> Tuple[np.ndarray, np.ndarray]:
        index, annotations = [], []
        for file_idx, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as handle:
                if self.h5_key not in handle:
                    raise KeyError(f"Dataset key '{self.h5_key}' not found in file: {path}")
                length = len(handle[self.h5_key])
                index.extend((file_idx, i) for i in range(length))

                if len(self.target_key):
                    annotations_json = handle[self.h5_annotations_key][()].decode()
                    ann_df = pd.DataFrame.from_dict(json.loads(annotations_json))
                    if len(ann_df) != length:
                        raise ValueError(f"Annotation length mismatch in {path}")

                    vals = ann_df[self.target_key].to_numpy()
                    if not np.issubdtype(vals.dtype, np.number):
                        vals = vals.astype(float)
                    annotations.extend(vals.tolist())

        return np.asarray(index, dtype=np.int64), np.asarray(annotations)

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def _get_handle(self, file_idx: int) -> h5py.File:
        handle = self._handles[file_idx]
        if handle is None:
            handle = h5py.File(self.h5_paths[file_idx], "r")
            self._handles[file_idx] = handle
        return handle

    def __getitem__(self, index: int):
        file_idx, image_idx = self._index[index]
        handle = self._get_handle(file_idx)
        image_np = np.asarray(handle[self.h5_key][image_idx], dtype=np.uint8)

        image = Image.fromarray(image_np, mode="L")

        if self.transforms:
            image = self.transforms(image)
        elif self.transform:
            image = self.transform(image)

        target = self.ann_idx[index] if len(self.target_key) else 0
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

        # --- Added for DINOv3 compatibility ---
    def get_target(self, index: int):
        """Return the integer class label for a single sample."""
        return int(self.ann_idx[index])

    def get_targets(self):
        """Return all labels as a numpy array."""
        return np.asarray(self.ann_idx, dtype=int)


    def __del__(self):
        for handle in self._handles:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
