"""Readable inference implementation of the five frozen PIVOT checkpoints.

Parameter names and operations match the archived evaluator. Only the inference
configuration used by these checkpoints is supported. In particular, the saved
encoder has TWO residual additions per block; changing that changes its outputs.
"""
from __future__ import annotations

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long).clamp(min=0, max=max_len)
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class PaddedConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, causal: bool):
        super().__init__()
        self.left_padding = kernel_size - 1 if causal else (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, self.right_padding)))


class GatedTemporalStem(nn.Module):
    def __init__(self, in_features: int, dim: int, kernels: list[int], dropout: float, causal: bool):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, dim)
        self.act = nn.GELU()
        self.branches = nn.ModuleList(
            PaddedConv1d(dim, k, causal)
            for k in kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(dim * len(kernels), 2 * dim, 1), nn.GLU(dim=1),
            nn.Dropout(dropout), nn.Conv1d(dim, dim, 1), nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.act(self.proj(self.norm(x)))
        base_t = base.transpose(1, 2)
        local = torch.cat([branch(base_t) for branch in self.branches], dim=1)
        mixed = self.mix(local).transpose(1, 2)
        return self.out_norm(base + mixed)


class ChannelMixer(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.out_proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(self.dropout(value * F.silu(gate))))


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, config: dict[str, Any]):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba3(
            d_model=dim, d_state=int(config['mamba_d_state']),
            headdim=int(config['mamba_headdim']), is_mimo=False,
            chunk_size=int(config['mamba_chunk_size']),
            is_outproj_norm=bool(config['mamba_outproj_norm']),
        )
        dropout = float(config['dropout'])
        self.fuse = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ChannelMixer(dim, int(dim * float(config['mlp_ratio'])), dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Stochastic depth is an identity during the checkpoint's evaluation.
        normed = self.norm1(x)
        mixed = self.fuse(torch.cat([self.mamba(normed)], dim=-1))
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class AttentiveStatsPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                   nn.Tanh(), nn.Linear(dim // 2, 1))
        self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x)
        scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        x_heads = x.unsqueeze(2)
        mean = (x_heads * weights).sum(dim=1)
        second = (x_heads.square() * weights).sum(dim=1)
        std = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        return torch.cat([mean, std], dim=-1).reshape(x.shape[0], -1)


class HeadProjection(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                 nn.Linear(in_dim, embed_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.LayerNorm(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CosineHead(nn.Module):
    def __init__(self, in_dim: int, config: dict[str, Any]):
        super().__init__()
        self.num_classes = int(config['num_classes'])
        embed_dim = int(config['head_embed_dim'])
        self.scale = float(config['cosine_scale'])
        self.embedding = HeadProjection(in_dim, embed_dim, float(config['dropout']))
        self.weight = nn.Parameter(torch.empty(self.num_classes, embed_dim))
        self.unknown_label = int(config['unknown_label'])
        self.unknown_logit_tau = float(config['unknown_logit_tau'])
        extra = int(config['unknown_class_centers']) - 1
        if extra > 0:
            self.unknown_extra_weight = nn.Parameter(torch.empty(extra, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = F.normalize(self.embedding(x), dim=-1)
        logits = F.linear(embedding, F.normalize(self.weight, dim=-1))
        if hasattr(self, 'unknown_extra_weight'):
            extras = F.linear(embedding, F.normalize(self.unknown_extra_weight, dim=-1))
            centers = torch.cat([logits[:, self.unknown_label:self.unknown_label + 1], extras], dim=1)
            tau = max(self.unknown_logit_tau, 1e-4)
            unknown = torch.logsumexp(centers / tau, dim=1) * tau
            logits = logits.clone()
            logits[:, self.unknown_label] = unknown.to(dtype=logits.dtype)
        return logits * self.scale


class PIVOT(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        required = {
            'architecture': 'flat', 'input_stem': 'gated_conv',
            'stem_stride': 1, 'channel_mixer': 'swiglu', 'pool_summary': 'attentive',
            'pool_heads': 1, 'pool_score_type': 'tanh', 'class_centers': 1,
            'adapter_dim': 0, 'bidirectional_mamba': False, 'local_evidence_mode': 'none',
            'local_attn_window': 0, 'chunk_context_weight': 0.0, 'mamba_is_mimo': False,
            'block_norm': 'layernorm', 'final_norm': 'layernorm',
            'stem_output_norm': 'layernorm', 'pool_score_norm': 'layernorm',
            'head_input_norm': 'layernorm', 'head_activation': 'gelu',
        }
        for name, expected in required.items():
            if config.get(name) != expected:
                raise ValueError(f'Unsupported inference setting {name}={config.get(name)!r}; expected {expected!r}')
        if config['stem_conv_mode'] not in ('causal', 'same'):
            raise ValueError('Unsupported stem convolution mode')
        dim = int(config['embed_dim'])
        self.input = GatedTemporalStem(len(config['feature_names']), dim,
                                       config['stem_kernels'], float(config['dropout']),
                                       config['stem_conv_mode'] == 'causal')
        self.blocks = nn.ModuleList(TemporalBlock(dim, config) for _ in range(int(config['depth'])))
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentiveStatsPool(dim)
        self.adapter = nn.Identity()
        self.head = CosineHead(2 * dim, config)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        tokens, lengths = batch
        mask = sequence_mask(lengths, tokens.shape[1])
        tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        x = self.input(tokens)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return self.head(self.adapter(self.pool(x, mask)))
