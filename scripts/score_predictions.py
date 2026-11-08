"""Score baseline logits aligned with a packaged held-out testset's row order."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from pivot_eval.artifacts import read_manifest, verify_scenario, prepare_testset
from pivot_eval.metrics import summarize_logits, summarize_open_world_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--scenario', choices=['closed_world','open_world','wtfpad','front','tamaraw'], required=True)
    p.add_argument('--logits', type=Path, required=True, help='NPY file, shape [test visits, classes], in frozen row order.')
    p.add_argument('--output', type=Path, default=Path('results/baseline_metrics.json'))
    a = p.parse_args()
    root = a.root.resolve()
    spec = read_manifest(root)['scenarios'][a.scenario]
    verify_scenario(root, spec)
    with pa.memory_map(str(prepare_testset(root, a.scenario, spec)), 'r') as source:
        labels = np.asarray(ipc.open_file(source).read_all()['label'].to_pylist(), dtype=np.int64)
    logits = np.load(a.logits, allow_pickle=False)
    if logits.shape != (len(labels), spec['num_classes']) or not np.isfinite(logits).all():
        raise ValueError('Expected finite [test visits, classes] logits in the frozen class and row order.')
    result = summarize_logits(labels, logits)
    if spec['unknown_label'] is not None:
        result.update(summarize_open_world_logits(labels, logits, unknown_label=spec['unknown_label']))
    result.update(scenario=a.scenario, num_samples=len(labels), source='external baseline predictions')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
