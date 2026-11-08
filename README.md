# PIVOT Evaluation

Evaluation code, pretrained weights, and testsets for PIVOT video fingerprinting
on traffic from video playback through Tor Browser. Training data are not included.

## Requirements

- Recommended GPU: **NVIDIA GeForce RTX 4090 (24 GB VRAM)**
- Linux x86-64 or WSL2, with an NVIDIA driver supporting CUDA 12.1
- Python 3.12 and [UV](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

Download and extract the repository, then open a terminal in its root directory.

```bash
uv sync --frozen
```

## Download weights and testsets

The accompanying `pivot-eval-assets-v1.tar` archive (about 276 MiB) contains
the weights and testsets for all five scenarios.

| Scenario | Test samples | Classes |
| --- | ---: | ---: |
| Closed world | 3,600 | 80 |
| Open world | 7,200 | 81 (including unknown) |
| WTF-PAD | 3,600 | 80 |
| FRONT | 3,600 | 80 |
| Tamaraw | 3,600 | 80 |

Place the provided archive in the repository root and install it:

```bash
uv run --frozen python scripts/manage_assets.py install --archive pivot-eval-assets-v1.tar
```

Verify the installed files:

```bash
uv run --frozen python run_evaluation.py --verify-only
```

## Run evaluation

Evaluate all five scenarios:

```bash
uv run --frozen python run_evaluation.py --scenario all --device cuda:0
```

For a single scenario, replace `all` with `closed_world`, `open_world`, `wtfpad`,
`front`, or `tamaraw`. For a quick run, add `--max-samples 16`.

Evaluate at fractions of each trace's duration:

```bash
uv run --frozen python run_evaluation.py --scenario closed_world --protocol paper-prefixes --device cuda:0
```

Evaluate at fixed elapsed times (seconds):

```bash
uv run --frozen python run_evaluation.py --scenario open_world --protocol elapsed --seconds 6 12 18 24 30 36 42 48 54 60 --device cuda:0
```

Results are saved to `results/<scenario>/<protocol>/` as `metrics.json` and
`metrics.csv`. Use `--output-dir results/my_run` for a separate output directory.
For less GPU memory, add `--batch-size 16`. Run with `--help` for all options.

Third-party licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
