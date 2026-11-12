"""Create a portable source-only submission zip; exclude runs, caches and links.

Usage: python package.py --output /path/to/SimpleButton_ICLR_Code.zip
The file hash manifest permits verification after transfer or extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(output):
    root = Path(__file__).resolve().parent
    excluded = {'runs', '__pycache__', '.pytest_cache', '.git', '.venv'}
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()
             and not set(p.relative_to(root).parts) & excluded
             and p.suffix not in {'.pyc', '.zip'} and p.name != 'SHA256SUMS.json']
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(root / 'SHA256SUMS.json')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, 'simplebutton_iclr_code/' + path.relative_to(root).as_posix())
    print(json.dumps(dict(archive=str(output), files=len(files), bytes=output.stat().st_size,
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output)
