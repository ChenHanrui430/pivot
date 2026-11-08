# Verification record

All five frozen checkpoints loaded with strict parameter matching. A readable
inference implementation was compared against the archived evaluator on four real
visits per scenario, at full observation, 6 seconds, and 10% duration. All 60 feature
comparisons were exactly equal; all 15 batch logit comparisons had maximum absolute
difference 0.0 in FP32. The checks confirm matching inference outputs on the inspected samples.

## Environment and execution checks

Verified on 2026-11-08 with Python 3.12.11, PyTorch 2.5.1+cu121, Triton 3.5.0,
FP32, and an RTX 3090 Ti. All five complete endpoint evaluations finished with
finite logits. The batches were 128 for closed world and defenses, and 48 for
open world; dynamic padding used multiples of 16.

## Frozen computation and evaluation scope

Each temporal block uses two residual additions. Closed/open checkpoints use
causal stem convolutions. Defense checkpoints use the same-padding stem setting
in their configs. Evaluation of a prefix reconstructs features only from packets
within that observation. These details preserve the archived checkpoints; replacing
their structure to match a different architectural description would change the
model being evaluated.

Open world here means direct 81-way classification. No validation-calibrated
first-decision risk policy, baseline model weights, training data, validation data,
or ablation experiments are supplied. One frozen primary/reference checkpoint is
included for each scenario. The artifact does not search or reselect checkpoints
using the held-out testsets.

All packet values and row ordering in the five necessary fields were checked
against the archived testsets before removing identifying metadata. Compressed
data, decompressed Arrow, checkpoint, and config hashes are in `manifest.json`.
The release archive uses a ten-file allowlist and normalized tar ownership/time.
The installer rejects extra files, traversal paths, duplicate entries, and links.

The upstream Mamba implementation files match their recorded source checksums.
See `verification.json` for machine-readable execution and parity checks.

A separate clean code checkout was synchronized with `uv sync --frozen`. The
release archive installed successfully; integrity checks passed for every
scenario. All five endpoint smoke evaluations (16 visits each), two absolute-time
open-world smoke observations (6/30 seconds), and two closed-world duration-prefix
smoke observations (0.1/1.0) completed successfully. The shared baseline scorer
reproduced the closed-world metrics when given the saved PIVOT logits.
