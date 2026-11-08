"""Build, download, or install only the testsets and tensor checkpoints in manifest.json."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pivot_eval.artifacts import artifact_path, read_manifest, sha256, verify_scenario

ASSET_NAME = 'pivot-eval-assets-v1.tar'


def records(root):
    return {spec[kind]['path']: spec[kind]
            for spec in read_manifest(root)['scenarios'].values()
            for kind in ('testset', 'checkpoint')}


def build(root, destination):
    for spec in read_manifest(root)['scenarios'].values():
        verify_scenario(root, spec)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / ASSET_NAME
    with tarfile.open(output, 'w', format=tarfile.USTAR_FORMAT) as archive:
        for relative, record in sorted(records(root).items()):
            path = artifact_path(root, relative)
            info = tarfile.TarInfo(relative)
            info.size = record['bytes']
            info.mode = 0o644
            info.mtime = info.uid = info.gid = 0
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    digest = sha256(output)
    (destination / f'{ASSET_NAME}.sha256').write_text(f'{digest}  {ASSET_NAME}\n')
    print(f'Built {output} ({output.stat().st_size:,} bytes)\nSHA256 {digest}')


def install(root, source):
    expected = records(root)
    # Validate all entries in a staging directory before replacing any artifacts.
    # extract()/extractall() are deliberately avoided: links and paths are untrusted.
    with tempfile.TemporaryDirectory(prefix='.asset-install-', dir=root) as temporary:
        staging = Path(temporary)
        seen = set()
        with tarfile.open(source, 'r:') as archive:
            for member in archive:
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise ValueError(f'Unexpected, duplicate, or non-file archive member: {member.name!r}')
                record = expected[member.name]
                if member.size != record['bytes']:
                    raise ValueError(f'Invalid member size: {member.name}')
                path = artifact_path(staging, member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, path.open('wb') as sink:
                    shutil.copyfileobj(stream, sink)
                if sha256(path) != record['sha256']:
                    raise ValueError(f'Checksum mismatch: {member.name}')
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f'Missing archive entries: {sorted(set(expected) - seen)}')
        for relative in sorted(expected):
            target = artifact_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, target)
    for spec in read_manifest(root)['scenarios'].values():
        verify_scenario(root, spec)
    print('Installed and verified all five testsets and five checkpoints.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--output-dir', type=Path, default=Path('release-assets'))
    installer = sub.add_parser('install')
    installer.add_argument('--archive', type=Path, required=True)
    downloader = sub.add_parser('download')
    downloader.add_argument('--repo', required=True, help='GitHub OWNER/REPOSITORY')
    downloader.add_argument('--tag', default='v1.0.0')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'build':
        build(root, args.output_dir.resolve())
    elif args.command == 'install':
        install(root, args.archive.resolve())
    else:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
            parser.error('--repo must be OWNER/REPOSITORY')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.tag):
            parser.error('Invalid release tag')
        url = f'https://github.com/{args.repo}/releases/download/{args.tag}/{ASSET_NAME}'
        handle, filename = tempfile.mkstemp(prefix='.download-', suffix='.tar', dir=root)
        try:
            expected_bytes = sum(r['bytes'] for r in records(root).values()) + 1024 * 1024
            with os.fdopen(handle, 'wb') as sink, urllib.request.urlopen(url, timeout=120) as response:
                size = 0
                while block := response.read(4 * 1024 * 1024):
                    size += len(block)
                    if size > expected_bytes:
                        raise ValueError('Downloaded archive exceeds the expected asset size.')
                    sink.write(block)
            install(root, Path(filename))
        finally:
            Path(filename).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
