# Uni-Core Source Snapshot for TripKa

This directory contains a snapshot of the customized Uni-Core Python source used in the `unipka` environment. Apply it over an official Uni-Core installation when deploying TripKa.

- `unicore/`: Complete core package source.
- `unicore_cli/`: Command-line entry points, including the customized training entry point.
- `manifest.json`: Source version, files modified relative to the installation RECORD, and SHA-256 checksums for all source files.
- `LICENSE`: License included with the installed package. Copyright notices in the source remain intact.

The original installation came from `unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl`, with package metadata version `0.0.1`. This snapshot excludes caches, bytecode, and compiled extensions. It is not a standalone pip package.

The following files differ from the original installation RECORD:

- `unicore/data/data_utils.py`
- `unicore/data/lmdb_dataset.py`
- `unicore/tasks/unicore_task.py`
- `unicore/trainer.py`
- `unicore_cli/train.py`

## Applying the Snapshot

Install official Uni-Core, its dependencies, and compiled extensions in the target environment first. Prefer the same Uni-Core version and Python / PyTorch / CUDA combination as the source installation; other combinations require separate compatibility validation. Then activate the target environment and run the following from the TripKa repository root:

```bash
conda activate unipka  # Replace with your deployment environment name
python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import json
import shutil

asset = Path('unicore').resolve()
manifest = json.loads((asset / 'manifest.json').read_text())
dist = distribution('unicore')  # Fails if the official package is not installed
target = Path(dist.locate_file('')).resolve()

# Validate the entire snapshot and target before copying any files.
for relative, expected in manifest['sha256'].items():
    source = asset / relative
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Source checksum mismatch: ' + relative)
for package in ('unicore', 'unicore_cli'):
    if not (target / package / '__init__.py').is_file():
        raise RuntimeError('Installed package not found: ' + str(target / package))
    if (target / package).resolve() == (asset / package).resolve():
        raise RuntimeError('The target must not be the source snapshot itself')
for relative in manifest['sha256']:
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(asset / relative, destination)
    # Remove stale bytecode so the next process uses the updated source.
    cache = destination.parent / '__pycache__'
    if cache.is_dir():
        for compiled in cache.glob(destination.stem + '.*.pyc'):
            compiled.unlink()
print('Copied', len(manifest['sha256']), 'source files to:', target)
PY
```

This operation preserves the official installation’s `.so` extensions and package metadata. It locates the target through installation metadata to avoid mistaking a repository directory for the installed package. Restart training or inference processes to use the updated source. Reinstalling the official package may overwrite these changes; apply the snapshot again afterward.
