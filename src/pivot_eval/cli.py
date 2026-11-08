from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .artifacts import read_manifest, verify_scenario, prepare_testset


def main():
    parser = argparse.ArgumentParser(description='Evaluate the frozen PIVOT checkpoints on held-out test sets.')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Artifact directory (default: current directory).')
    parser.add_argument('--scenario', choices=['all', 'closed_world', 'open_world', 'wtfpad', 'front', 'tamaraw'], default='all')
    parser.add_argument('--protocol', choices=['endpoint', 'paper-prefixes', 'elapsed'], default='endpoint')
    parser.add_argument('--ratios', type=float, nargs='+', help='Paper-prefix fractions of each retained trace duration, not absolute seconds.')
    parser.add_argument('--seconds', type=float, nargs='+', help='Absolute elapsed-time checkpoints; requires --protocol elapsed.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--max-samples', type=int, default=0, help='Smoke test only; 0 evaluates the entire test split.')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--save-logits', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1 or args.workers < 0 or args.max_samples < 0:
        parser.error('Batch size must be positive; workers and max-samples must be nonnegative.')
    if args.ratios and args.protocol != 'paper-prefixes' or args.seconds and args.protocol != 'elapsed':
        parser.error('--ratios requires paper-prefixes; --seconds requires elapsed.')
    if args.protocol == 'paper-prefixes':
        values = args.ratios or [i / 10 for i in range(1, 11)]
        if any(not 0 < x <= 1 for x in values): parser.error('Ratios must lie in (0, 1].')
        observations = [('duration_ratio', v) for v in values]
    elif args.protocol == 'elapsed':
        values = args.seconds or list(range(6, 61, 6))
        if any(not 0 < x <= 60 for x in values): parser.error('Seconds must lie in (0, 60].')
        observations = [('time', v) for v in values]
    else:
        observations = [('full', None)]
    root = args.root.resolve()
    manifest = read_manifest(root)
    scenarios = manifest['scenarios']
    selected = list(scenarios) if args.scenario == 'all' else [args.scenario]
    for name in selected:
        spec = scenarios[name]
        verify_scenario(root, spec)
        print(f'[verified] {name}: {spec["testset"]["rows"]} test visits', flush=True)
        if args.verify_only: continue
        arrow = prepare_testset(root, name, spec)
        from .inference import evaluate
        import numpy as np
        config = json.loads((root / spec['config']['path']).read_text())
        rows, logits, labels = evaluate(arrow, root / spec['checkpoint']['path'], config, spec,
            observations, device=args.device, batch_size=args.batch_size or spec['reference_batch_size'],
            workers=args.workers, max_samples=args.max_samples, precision=args.precision)
        target = args.output_dir / name / args.protocol
        target.mkdir(parents=True, exist_ok=True)
        for row in rows:
            row['scenario'] = name
        (target / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
        columns = list(rows[0])
        with (target / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
        if args.save_logits:
            np.savez_compressed(target / 'predictions.npz', logits=logits, labels=labels,
                slice_modes=np.asarray([m for m, _ in observations]),
                slice_values=np.asarray([v if v is not None else np.nan for _, v in observations]))
        print(f'[saved] {target / "metrics.json"}', flush=True)


if __name__ == '__main__':
    main()
