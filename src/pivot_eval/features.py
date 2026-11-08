from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Literal
import numpy as np
TokenMode = Literal["countmamba"]
SliceMode = Literal["full", "time", "duration_ratio"]


@dataclass(frozen=True)
class BurstFeatureConfig:
    max_time: float = 60.0
    packet_size_scale: float = 1500.0
    max_tokens: int = 2048
    token_mode: TokenMode = "burst"
    window_packets: int = 64
    window_stride: int = 32
    use_effective_only: bool = True
    count_seq_len: int = 25000
    count_cell_size: int = 512
    count_max_cell: int = 2
    count_time_threshold: float = 0.1
    count_bin_width: float = 0.0
    count_log_transform: bool = True

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)

def countmamba_feature_names(max_cell: int = 2) -> tuple[str, ...]:
    names: list[str] = []
    for cell in range(max(int(max_cell), 0) + 1):
        names.extend((f"cell_{cell}_in_count", f"cell_{cell}_out_count"))
    names.extend(("time_bin_gap", "cluster_count"))
    return tuple(names)

def slice_trace(
    timestamps: np.ndarray,
    signed_sizes: np.ndarray,
    payload_sizes: np.ndarray,
    is_effective: np.ndarray | None,
    config: BurstFeatureConfig,
    mode: SliceMode = "full",
    value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = timestamps.astype(np.float32, copy=False)
    signed_sizes = signed_sizes.astype(np.float32, copy=False)
    payload_sizes = payload_sizes.astype(np.float32, copy=False)
    if is_effective is not None and config.use_effective_only:
        effective_mask = is_effective.astype(bool, copy=False)
        timestamps = timestamps[effective_mask]
        signed_sizes = signed_sizes[effective_mask]
        payload_sizes = payload_sizes[effective_mask]

    valid = (timestamps >= 0.0) & (timestamps <= float(config.max_time)) & (signed_sizes != 0.0)
    timestamps = timestamps[valid]
    signed_sizes = signed_sizes[valid]
    payload_sizes = payload_sizes[valid]

    if mode == "full":
        return timestamps, signed_sizes, payload_sizes
    if value is None:
        raise ValueError(f"slice value is required for slice_mode={mode!r}")
    if mode == "time":
        mask = timestamps <= max(float(value), 0.0)
        return timestamps[mask], signed_sizes[mask], payload_sizes[mask]
    if mode == "ratio":
        ratio = float(value)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio slice value must be in (0, 1]")
        if timestamps.size == 0:
            return timestamps, signed_sizes, payload_sizes
        cutoff = int(np.ceil(timestamps.size * ratio))
        return timestamps[:cutoff], signed_sizes[:cutoff], payload_sizes[:cutoff]
    if mode == "duration_ratio":
        ratio = float(value)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("duration_ratio slice value must be in (0, 1]")
        if timestamps.size == 0:
            return timestamps, signed_sizes, payload_sizes
        cutoff_time = float(timestamps[-1]) * ratio
        cutoff = int(np.searchsorted(timestamps, cutoff_time, side="right"))
        return timestamps[:cutoff], signed_sizes[:cutoff], payload_sizes[:cutoff]
    raise ValueError(f"Unsupported slice_mode: {mode!r}")

def _build_countmamba_tokens(
    ts: np.ndarray,
    signed: np.ndarray,
    config: BurstFeatureConfig,
) -> tuple[np.ndarray, int]:
    max_tokens = max(int(config.max_tokens), 1)
    max_cell = max(int(config.count_max_cell), 0)
    cell_size = max(float(config.count_cell_size), 1.0)
    seq_len = max(int(config.count_seq_len), 0)
    if seq_len > 0 and signed.size > seq_len:
        ts = ts[:seq_len]
        signed = signed[:seq_len]

    feature_dim = len(countmamba_feature_names(max_cell))
    tokens = np.zeros((max_tokens, feature_dim), dtype=np.float32)
    if ts.size == 0:
        return tokens, 0

    bin_width = float(config.count_bin_width) if float(config.count_bin_width) > 0.0 else float(config.max_time) / float(max_tokens)
    if bin_width <= 0.0:
        raise ValueError("count bin width must be positive for countmamba token mode")
    time_interval = bin_width * float(config.count_time_threshold)

    current_index = 0
    current_timestamps: list[float] = []
    last_index = 0
    for signed_value, timestamp in zip(signed, ts):
        direction = int(np.sign(signed_value))
        if direction == 0:
            continue
        cell = min(int(abs(float(signed_value)) // cell_size), max_cell)
        fragment = 0 if direction < 0 else 1
        channel = 2 * cell + fragment
        index = min(math.floor(float(timestamp) / bin_width), max_tokens - 1)
        index = max(index, 0)
        tokens[index, channel] += 1.0
        last_index = max(last_index, index)

        if index != current_index:
            tokens[index, 2 * max_cell + 2] = max(index - current_index, 0)
            if current_timestamps:
                delta_t = np.diff(np.asarray(current_timestamps, dtype=np.float32))
                cluster_count = int(np.sum(delta_t > time_interval) + 1)
            else:
                cluster_count = 0
            tokens[current_index, 2 * max_cell + 3] = float(cluster_count)
            current_index = index
            current_timestamps = [float(timestamp)]
        else:
            current_timestamps.append(float(timestamp))

    if current_timestamps:
        delta_t = np.diff(np.asarray(current_timestamps, dtype=np.float32))
        cluster_count = int(np.sum(delta_t > time_interval) + 1)
        tokens[current_index, 2 * max_cell + 3] = float(cluster_count)

    if config.count_log_transform:
        tokens = np.log1p(tokens, dtype=np.float32)
    return tokens.astype(np.float32, copy=False), int(last_index + 1)

def feature_names_for_config(config: BurstFeatureConfig) -> tuple[str, ...]:
    return countmamba_feature_names(config.count_max_cell)


def build_tokens(timestamps, signed_sizes, payload_sizes, is_effective, config,
                 slice_mode="full", slice_value=None):
    ts, signed, _ = slice_trace(timestamps, signed_sizes, payload_sizes, is_effective,
                                config, slice_mode, slice_value)
    return _build_countmamba_tokens(ts, signed, config)
