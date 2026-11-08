"""Integrity checking and unpacking; usable without importing CUDA libraries."""
from __future__ import annotations
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['training_data_included'] or manifest['validation_data_included']:
        raise ValueError('This artifact must contain only test data.')
    return manifest


def artifact_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe artifact path: {relative}')
    return path


def verify_scenario(root: Path, spec: dict) -> None:
    for kind in ['testset', 'checkpoint', 'config']:
        record = spec[kind]
        path = artifact_path(root, record['path'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing {record["path"]}; download the release assets first.')
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Integrity check failed: {record["path"]}')


def prepare_testset(root: Path, name: str, spec: dict) -> Path:
    output = root / '.cache' / 'testsets' / f'{name}.arrow'
    if output.is_file() and sha256(output) == spec['testset']['uncompressed_sha256']:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=name, suffix='.partial', dir=output.parent)
    try:
        with os.fdopen(handle, 'wb') as sink, lzma.open(root / spec['testset']['path'], 'rb') as source:
            shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
        if sha256(Path(temp)) != spec['testset']['uncompressed_sha256']:
            raise ValueError(f'Uncompressed testset hash mismatch: {name}')
        os.replace(temp, output)
    finally:
        Path(temp).unlink(missing_ok=True)
    return output
