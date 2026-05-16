#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking}"
MODEL_FILE="$MODEL_DIR/model.safetensors"
URL="${URL:-https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/resolve/main/model.safetensors}"
TOTAL_BYTES="${TOTAL_BYTES:-5336011242}"
CHUNKS="${CHUNKS:-16}"
PART_DIR="${PART_DIR:-$MODEL_DIR/.parallel_parts}"
META_FILE="$PART_DIR/download.meta"

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_FILE" ]]; then
  : > "$MODEL_FILE"
fi

start="$(stat -c '%s' "$MODEL_FILE")"
if (( start > TOTAL_BYTES )); then
  echo "Existing file is larger than expected: $start > $TOTAL_BYTES" >&2
  exit 1
fi
if (( start == TOTAL_BYTES )); then
  echo "already complete: $MODEL_FILE ($start bytes)"
  exit 0
fi

mkdir -p "$PART_DIR"

remain=$((TOTAL_BYTES - start))
base=$((remain / CHUNKS))
if (( base <= 0 )); then
  CHUNKS=1
  base=$remain
fi

echo "resume_prefix_bytes=$start"
echo "remaining_bytes=$remain"
echo "chunks=$CHUNKS"

if [[ -f "$META_FILE" ]]; then
  if ! grep -qx "start=$start" "$META_FILE" \
    || ! grep -qx "total=$TOTAL_BYTES" "$META_FILE" \
    || ! grep -qx "chunks=$CHUNKS" "$META_FILE"; then
    echo "Existing part metadata does not match this run:" >&2
    cat "$META_FILE" >&2
    echo "Refusing to reuse possibly incompatible parts." >&2
    exit 1
  fi
else
  {
    echo "start=$start"
    echo "total=$TOTAL_BYTES"
    echo "chunks=$CHUNKS"
  } > "$META_FILE"
fi

download_part() {
  local i="$1"
  local s="$2"
  local e="$3"
  local out="$4"
  local expected actual sub_start tail_expected tmp got

  expected=$((e - s + 1))
  if [[ -f "$out" ]]; then
    actual="$(stat -c '%s' "$out")"
  else
    actual=0
    : > "$out"
  fi

  if (( actual == expected )); then
    echo "chunk $i: already complete ($actual / $expected)"
    return 0
  fi
  if (( actual > expected )); then
    echo "chunk $i: part larger than expected ($actual > $expected)" >&2
    return 1
  fi

  sub_start=$((s + actual))
  tail_expected=$((expected - actual))
  tmp="$out.tail.$$"
  rm -f "$tmp"
  echo "chunk $i: resume bytes $sub_start-$e ($actual / $expected already)"
  curl -sS -L --fail --retry 20 --retry-delay 5 --connect-timeout 30 \
    -r "$sub_start-$e" \
    -o "$tmp" \
    "$URL"
  got="$(stat -c '%s' "$tmp")"
  if (( got != tail_expected )); then
    echo "chunk $i: tail size mismatch ($got / $tail_expected)" >&2
    rm -f "$tmp"
    return 1
  fi
  cat "$tmp" >> "$out"
  rm -f "$tmp"
  actual="$(stat -c '%s' "$out")"
  echo "chunk $i: complete ($actual / $expected)"
}

pids=()
for i in $(seq 0 $((CHUNKS - 1))); do
  s=$((start + i * base))
  if (( i == CHUNKS - 1 )); then
    e=$((TOTAL_BYTES - 1))
  else
    e=$((start + (i + 1) * base - 1))
  fi
  out="$PART_DIR/part_${i}.bin"
  download_part "$i" "$s" "$e" "$out" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

for i in $(seq 0 $((CHUNKS - 1))); do
  s=$((start + i * base))
  if (( i == CHUNKS - 1 )); then
    e=$((TOTAL_BYTES - 1))
  else
    e=$((start + (i + 1) * base - 1))
  fi
  expected=$((e - s + 1))
  actual="$(stat -c '%s' "$PART_DIR/part_${i}.bin")"
  echo "part $i size: $actual / $expected"
  if (( actual != expected )); then
    echo "size mismatch for part $i" >&2
    exit 1
  fi
done

for i in $(seq 0 $((CHUNKS - 1))); do
  cat "$PART_DIR/part_${i}.bin" >> "$MODEL_FILE"
done

final_size="$(stat -c '%s' "$MODEL_FILE")"
echo "final size: $final_size / $TOTAL_BYTES"
if (( final_size != TOTAL_BYTES )); then
  echo "final size mismatch" >&2
  exit 1
fi

rm -rf "$PART_DIR"
sha256sum "$MODEL_FILE" | tee "$MODEL_FILE.sha256"
