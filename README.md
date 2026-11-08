# PIVOT Evaluation

Evaluation code, pretrained weights, and testsets for PIVOT video fingerprinting
on traffic from video playback through Tor Browser. Training data are not included.

## Requirements

- Recommended GPU: **NVIDIA GeForce RTX 4090 (24 GB VRAM)**
- Linux x86-64 or WSL2, with an NVIDIA driver supporting CUDA 12.1
- Python 3.12 and [UV](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

```bash
git clone https://github.com/ChenHanrui430/pivot.git
cd pivot
uv sync --frozen
```

## Download weights and testsets

[Release v1.0.0](https://github.com/ChenHanrui430/pivot/releases/tag/v1.0.0)
provides the weights and testsets in `pivot-eval-assets-v1.tar` (about 276 MiB).

| Scenario | Test samples | Classes |
| --- | ---: | ---: |
| Closed world | 3,600 | 80 |
| Open world | 7,200 | 81 (including unknown) |
| WTF-PAD | 3,600 | 80 |
| FRONT | 3,600 | 80 |
| Tamaraw | 3,600 | 80 |

Download and install all five weights and testsets:

```bash
uv run --frozen python scripts/manage_assets.py download --repo ChenHanrui430/pivot --tag v1.0.0
```

Alternatively, download the archive from the Release page and install it locally:

```bash
uv run --frozen python scripts/manage_assets.py install --archive /path/to/pivot-eval-assets-v1.tar
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
