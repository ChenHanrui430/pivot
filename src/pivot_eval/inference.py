from __future__ import annotations
import functools
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .dataset import TestDataset, collate
from .metrics import summarize_logits, summarize_open_world_logits
from .model import PIVOT


@torch.inference_mode()
def evaluate(arrow: Path, checkpoint: Path, config: dict, spec: dict,
             observations: list[tuple[str, float | None]], *, device: str,
             batch_size: int, workers: int, max_samples: int, precision: str):
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):
        raise RuntimeError('The packaged Mamba-3 SISO kernels require an NVIDIA CUDA GPU.')
    # Triton uses the current CUDA device when selecting/compiling its kernels.
    torch.cuda.set_device(torch.device(device))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    model = PIVOT(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    rows, saved_logits, saved_labels = [], [], None
    for mode, value in observations:
        dataset = TestDataset(arrow, config, mode, value)
        if len(dataset) != spec['testset']['rows']:
            raise ValueError('Testset row count differs from the frozen manifest.')
        if max_samples > 0:
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=workers, collate_fn=functools.partial(collate, pad_to_multiple=16),
                            pin_memory=True, persistent_workers=workers > 0)
        logits, labels = [], []
        start = time.perf_counter()
        for (tokens, lengths), y in tqdm(loader, desc=f'{mode}={value}', unit='batch'):
            x = tokens.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=precision == 'bf16', dtype=torch.bfloat16):
                out = model(x)
            logits.append(out.float().cpu().numpy())
            labels.append(y.numpy())
        yy, zz = np.concatenate(labels), np.concatenate(logits)
        if not np.all(np.isfinite(zz)):
            raise RuntimeError('Non-finite logits; inference results are invalid.')
        if saved_labels is not None and not np.array_equal(saved_labels, yy):
            raise RuntimeError('Test sample ordering changed across observations.')
        saved_labels = yy
        result = summarize_logits(yy, zz)
        if spec['unknown_label'] is not None:
            result.update(summarize_open_world_logits(yy, zz, unknown_label=spec['unknown_label']))
        result.update({'num_samples': len(yy), 'slice_mode': mode, 'slice_value': value,
                       'batch_size': batch_size, 'precision': precision,
                       'evaluation_seconds': round(time.perf_counter() - start, 3),
                       'smoke_test': max_samples > 0})
        rows.append(result)
        saved_logits.append(zz)
    return rows, np.stack(saved_logits), saved_labels
