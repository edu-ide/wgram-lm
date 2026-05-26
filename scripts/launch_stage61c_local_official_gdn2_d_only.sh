#!/usr/bin/env bash
set -euo pipefail

echo "Deprecated name: this is not attention-free GDN2-only."
echo "Forwarding to the official GDN2-backed Qwen3.5-style 3:1 launcher."

exec "$(dirname "$0")/launch_stage61c_local_official_gdn2_3to1.sh" "$@"
