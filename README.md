# PIVOT: anonymous evaluation artifact

PIVOT identifies the exact video played on a website accessed through Tor Browser,
using observed packet directions, sizes, and relative timing. This repository
contains readable inference code, held-out testsets, and one frozen trained
checkpoint per scenario. It supports evaluation without training data.

## Included evaluations

| Scenario | Test visits | Output labels | Checkpoint |
| --- | ---: | --- | --- |
| Closed world | 3,600 | 80 monitored videos, IDs 0–79 | One frozen model |
| Direct open world | 7,200 | IDs 0–79 plus unknown ID 80 | One frozen model |
| WTF-PAD | 3,600 | 80 monitored videos | One defense-trained model |
| FRONT | 3,600 | 80 monitored videos | One defense-trained model |
| Tamaraw | 3,600 | 80 monitored videos | One defense-trained model |

Each monitored class has 45 held-out visits. The open-world set contains 3,600
monitored visits and 3,600 unmonitored visits. Training/validation sets, collection
scripts, training entry points, ablations, and repeated-seed experiments are not
included. The frozen checkpoints are the primary/reference run's selected models;
no checkpoint selection is performed on the released testsets.

## Requirements and installation

Use Linux x86-64 (including WSL2), Python 3.12, UV, and an NVIDIA CUDA GPU.
The lockfile uses PyTorch 2.5.1 with CUDA 12.1 and Triton 3.5.0, matching the
source evaluation environment. Use an up-to-date NVIDIA driver compatible with
CUDA 12.1. CPU, macOS, and Windows-native inference are not supported by these
Mamba-3 SISO kernels. An RTX 3090 Ti (24 GB) is used for local verification.
Smaller batches can reduce memory demand but may produce numerical differences.
Allow approximately 10 GB for the Python environment, assets, extracted testsets,
and caches. Initial kernel compilation takes longer than later evaluations.

Install UV if needed, following https://docs.astral.sh/uv/getting-started/installation/:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Open a new shell if uv is not yet on PATH.
uv sync --frozen
```

No separate Mamba installation, CUDA extension build, TileLang, or causal-conv1d
package is needed. The pinned Mamba-3 SISO source subset is included with its
upstream license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Obtain and verify testsets and weights

Clone the repository and run commands from its root. Code/configuration and the
manifest are in Git. The ten binary files are provided as the release asset
`pivot-eval-assets-v1.tar` (about 276 MiB). Replace `OWNER/REPOSITORY` with the
repository hosting this artifact:

```bash
uv run --frozen python scripts/manage_assets.py download --repo OWNER/REPOSITORY --tag v1.0.0
uv run --frozen python run_evaluation.py --scenario all --verify-only
```

If you downloaded the release asset manually, install it with:

```bash
uv run --frozen python scripts/manage_assets.py install --archive /path/to/pivot-eval-assets-v1.tar
```

Installation accepts only manifest-listed testset and checkpoint files, checks
size and SHA256, and rejects extra files, links, duplicate names, or unsafe paths.
Decompressed Arrow files are cached locally and checked against their own hashes.
The repository's prepared local copy already has the assets installed.

## Run evaluation

A small smoke test checks all five checkpoint/data pairs:

```bash
uv run --frozen python run_evaluation.py --scenario all --device cuda:0 --max-samples 16
```

Full endpoint evaluation (the entire retained observation, up to 60 seconds):

```bash
uv run --frozen python run_evaluation.py --scenario all --device cuda:0 --workers 4
```

Use `--scenario closed_world`, `open_world`, `wtfpad`, `front`, or `tamaraw`
for one scenario. Default batches are 128 (closed/defenses) and 48 (open).
The reference precision is FP32. Optional `--precision bf16` is for exploratory
runs; it is not the reference protocol. For limited GPU memory, try
`--batch-size 16`. Use `--workers 0` if worker processes are unavailable.

Paper prefix reporting uses a fraction of each retained trace's duration:

```bash
uv run --frozen python run_evaluation.py --scenario closed_world --protocol paper-prefixes --device cuda:0
```

These duration fractions use the final timestamp and are an **offline reporting
protocol**. They must not be described as fixed elapsed seconds or a deployable
online stopping rule. Absolute elapsed-time evaluation truncates packets before
feature extraction and does not use future timestamps to set the cutoff:

```bash
uv run --frozen python run_evaluation.py --scenario open_world --protocol elapsed --seconds 6 12 18 24 30 36 42 48 54 60 --device cuda:0
```

Open-world inference here is direct 81-way classification, including the unknown
class. It is not the paper's validation-calibrated first-decision risk policy;
policy thresholds and policy replay are outside this reduced artifact's scope.
Defense testsets are already transformed; no defense simulation or retraining is
required to evaluate their paired checkpoints.

Metrics go to `results/SCENARIO/PROTOCOL/metrics.json` and `metrics.csv`.
`--save-logits` also writes `predictions.npz` containing ordered labels, logits,
and observation settings. Smoke-test results are explicitly marked and must not
be compared to full-test scores. Reusing an output directory replaces previous
results; pass `--output-dir results/my_run` to retain a separate run.

## Paper scores and implementation

`paper_reference_metrics.json` records endpoint scores reported in the paper,
in percent. The evaluation commands write the metrics computed for each run.
Installation and implementation checks are documented in
[docs/verification.md](docs/verification.md).

The source implementation preserves the frozen model's actual operations:
**each temporal block has two residual additions**, one after Mamba/fusion and one
after the channel mixer. Changing that computation would invalidate comparison
with these weights. The closed/direct-open checkpoints use a causal stem; the
archived defense checkpoints use same-padding stems, as recorded in their
configs. Each prefix evaluation rebuilds inputs from only the observed packets. Dropout is disabled during evaluation. All checkpoint tensors
are loaded with `weights_only=True` and strict parameter matching.

## Data representation and anonymity

The testsets contain Apache Arrow IPC files compressed with XZ. Only these fields
are retained:

| Field | Meaning |
| --- | --- |
| `timestamps` | Per-packet relative arrival time in seconds |
| `signed_sizes` | Signed packet size encoding direction |
| `payload_sizes` | Per-packet payload size |
| `is_effective` | Effective-packet filter flag |
| `label` | Anonymous video class ID; 80 is unknown only in open world |

Packet arrays retain their original order and values. Schema metadata, source
paths, session/region identifiers, site/video names, and label-to-URL mappings are
not included. These are traffic-derived observations, not video files or raw
PCAPs. Only held-out test visits are distributed. The WTCM feature construction
and all inference parameters are included; see `manifest.json` and the five
`checkpoints/*/config.json` files.

## Optional baseline scoring

No baseline weights are bundled. If an external baseline produces finite logits
in the exact test-row order and class order, use the shared scorer:

```bash
uv run --frozen python scripts/score_predictions.py --scenario closed_world --logits /path/to/baseline_logits.npy --output results/baseline_closed.json
```

The NPY array must have shape `[3600,80]`, or `[7200,81]` for open world.
Use the packet arrays with the baseline's own documented preprocessing; PIVOT's
WTCM preprocessing is not automatically appropriate for other methods. Test data
must not be used to fit or tune a baseline.
