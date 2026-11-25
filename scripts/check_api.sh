#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
# shellcheck disable=SC1091
source "$script_dir/api_env.sh"
vision_avoid_load_api_env

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /path/to/scene.jpg\n' "$0" >&2
    exit 2
fi
image_path="$1"
if [[ ! -r "$image_path" ]]; then
    printf 'Image is missing or unreadable: %s\n' "$image_path" >&2
    exit 2
fi

cd -- "$project_dir"
PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - "$image_path" <<'PY'
import sys
import time
from pathlib import Path

from vision_avoid.config import load_config
from vision_avoid.providers import create_scanner

cfg = load_config("config/indoor.yaml")
if cfg.vlm.provider != "api" or not cfg.vlm.api_enabled:
    raise SystemExit("API is not enabled in the loaded configuration")
scanner = create_scanner(cfg.vlm)
started = time.monotonic()
result = scanner.scan(Path(sys.argv[1]).read_bytes())
elapsed = time.monotonic() - started
print(result.model_dump_json(indent=2))
print(
    f"provider={cfg.vlm.provider} model={cfg.vlm.api_model} "
    f"elapsed={elapsed:.2f}s usage={scanner.last_usage}"
)
PY
