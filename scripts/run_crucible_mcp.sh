#!/bin/zsh
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_root"
exec "$repo_root/.venv/bin/python3" -m crucible.mcp.server
