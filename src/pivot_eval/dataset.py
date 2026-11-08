"""Read the held-out packet metadata; construct prefixes before WTCM features."""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import math
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import Dataset
from .features import BurstFeatureConfig, build_tokens


def feature_config(config: dict) -> BurstFeatureConfig:
    raw = config['feature_config']
    valid = {f.name for f in fields(BurstFeatureConfig)}
    return BurstFeatureConfig(**{k: v for k, v in raw.items() if k in valid})


class TestDataset(Dataset):
    def __init__(self, path: Path, config: dict, mode: str, value: float | None):
        self.path = path
        self.config = feature_config(config)
        self.mode, self.value = mode, value
        self._source = None
        self._table = None
        self._open()
        self.length = self._table.num_rows

    def _open(self):
        if self._table is None:
            self._source = pa.memory_map(str(self.path), 'r')
            self._table = ipc.open_file(self._source).read_all()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_source'] = state['_table'] = None
        return state

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open()
        table = self._table
        ts = np.asarray(table['timestamps'][index].as_py(), dtype=np.float32)
        signed = np.asarray(table['signed_sizes'][index].as_py(), dtype=np.float32)
        payload = np.asarray(table['payload_sizes'][index].as_py(), dtype=np.float32)
        effective = np.asarray(table['is_effective'][index].as_py(), dtype=bool)
        x, length = build_tokens(ts, signed, payload, effective, self.config,
                                 slice_mode=self.mode, slice_value=self.value)
        label = int(table['label'][index].as_py())
        return (torch.from_numpy(x), torch.tensor(length, dtype=torch.long)), torch.tensor(label)


def collate(batch, pad_to_multiple: int = 16):
    features, lengths, labels = [], [], []
    max_len = 1
    for (x, length), y in batch:
        max_len = max(max_len, min(max(int(length), 1), x.shape[0]))
        features.append(x)
        lengths.append(length)
        labels.append(y)
    if pad_to_multiple > 1:
        max_len = min(math.ceil(max_len / pad_to_multiple) * pad_to_multiple, features[0].shape[0])
    return (torch.stack([x[:max_len] for x in features]), torch.stack(lengths)), torch.stack(labels)
