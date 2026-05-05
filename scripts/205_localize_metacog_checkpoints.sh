#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"

BASELINE_SOURCE="${BASELINE_SOURCE:-runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-runs/qwen35_2b_4090_metacog_unknown_teacher_kl_conservative_s040/last.pt}"
LOCAL_CKPT_ROOT="${LOCAL_CKPT_ROOT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_fusion_sweep}"

BASELINE_DEST="$LOCAL_CKPT_ROOT/no_warmup_s001/last.pt"
CANDIDATE_DEST="$LOCAL_CKPT_ROOT/unknown_teacher_kl_conservative_s040/last.pt"

check_writable_dir() {
  local path="$1"
  mkdir -p "$path"
  "$PYTHON" - "$path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
probe = path / "preflight_write_test"
try:
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
except Exception as exc:
    raise SystemExit(
        f"Output directory is not writable: {path}\n"
        f"{type(exc).__name__}: {exc}"
    )
PY
}

localize_checkpoint() {
  local label="$1"
  local src="$2"
  local dest="$3"
  mkdir -p "$(dirname "$dest")"
  "$PYTHON" - "$label" "$src" "$dest" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

label, src_arg, dest_arg = sys.argv[1], sys.argv[2], sys.argv[3]
src = Path(src_arg)
dest = Path(dest_arg)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

try:
    if not src.is_file():
        raise FileNotFoundError(src)
    open(src, 'rb').read(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    src_hash = sha256(src)
    dest_hash = sha256(dest)
    if src_hash != dest_hash:
        raise RuntimeError(f"sha256 mismatch: {src_hash} != {dest_hash}")
except Exception as exc:
    raise SystemExit(
        f"Failed to localize {label}: {src} -> {dest}\n"
        f"{type(exc).__name__}: {exc}"
    )

print(f"{label}: {dest}")
print(f"{label}_sha256: {dest_hash}")
PY
}

check_writable_dir "$LOCAL_CKPT_ROOT"
localize_checkpoint "baseline" "$BASELINE_SOURCE" "$BASELINE_DEST"
localize_checkpoint "candidate" "$CANDIDATE_SOURCE" "$CANDIDATE_DEST"

cat <<EOF

Use these paths for the fusion sweep:

BASELINE_CHECKPOINT=$BASELINE_DEST \\
CANDIDATE_CHECKPOINT=$CANDIDATE_DEST \\
bash scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
EOF
