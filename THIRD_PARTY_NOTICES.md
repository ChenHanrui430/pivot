The `src/mamba_ssm` subtree contains the Mamba-3 SISO module and its Triton
kernel dependencies from https://github.com/state-spaces/mamba at commit
`33e2849ff046f72d60db8b17dbf47b7c949ff8c1`.

The eight upstream implementation files are unchanged. Minimal package
`__init__.py` files restrict imports to this inference path. Original copyright
headers are preserved. The upstream Apache License 2.0 is reproduced in
`third_party/MAMBA_LICENSE`; source checksums are in
`third_party/mamba_source.json`. Optional upstream MIMO/TileLang/CuTe imports are
not used by the supplied SISO checkpoints. Backward kernel definitions remain
because the upstream combined module imports them; this artifact supplies no
training entry point.

The license above applies to the upstream Mamba files. This evaluation artifact
does not assign a separate open-source or data license to the remaining code,
weights, or testsets. Public availability for review is not a grant of unrestricted
redistribution rights. Training and validation data are not part of this release.
