#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${RISK_PYTHON:-python}"
mode="${1:-isolated}"
if [[ "$mode" != isolated && "$mode" != --image-torch ]]; then
    printf 'Usage: bash tools/install.sh [--image-torch]\n' >&2; exit 2
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
requirements="$ROOT/requirements.lock"
doctor_args=()
if [[ "$mode" == --image-torch ]]; then
    "$PYTHON" -c 'import torch; assert torch.__version__.split("+")[0] == "2.3.1" and torch.version.cuda == "12.1", "Image must supply torch 2.3.1 / CUDA 12.1"; assert torch.cuda.is_available(), "GPU unavailable"; print("Keeping existing", torch.__version__, "CUDA", torch.version.cuda)'
    mkdir -p "$ROOT/.cache"
    "$PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
lines = (root / 'requirements.lock').read_text().splitlines()
lines = [s for s in lines if not s.startswith(('torch==', 'triton==', 'nvidia-', '--extra-index-url'))]
(root / '.cache/requirements.image.lock').write_text('\n'.join(lines) + '\n')
PY
    requirements="$ROOT/.cache/requirements.image.lock"
    doctor_args+=(--image-torch)
else
    "$PYTHON" -c 'import os,sys; assert sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX"), "Activate a dedicated venv/conda environment first, or use --image-torch"'
fi
"$PYTHON" -m pip install --index-url https://pypi.org/simple pip==25.3 setuptools==80.9.0 wheel==0.45.1
# --no-deps preserves the GPU simulator despite old Safety-Gymnasium metadata.
"$PYTHON" -m pip install --index-url https://pypi.org/simple --no-deps -r "$requirements"
"$PYTHON" "$ROOT/tools/doctor.py" --versions-only "${doctor_args[@]}"
printf '\nInstall complete. Run: python tools/doctor.py --gpu 0 %s\n' "${doctor_args[*]}"
