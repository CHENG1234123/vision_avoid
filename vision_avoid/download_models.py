from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


HF_MODELS = {
    "dfine-small-coco": (
        "ustc-community/dfine-small-coco",
        "f79e65b5fbb33ceb9d3ebba042955d7410c608f8",
    ),
    "depth-anything-v2-metric-indoor-small-hf": (
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "8078d68a9c75a972131914f6afd0c1723be0da7f",
    ),
    "depth-anything-v2-metric-outdoor-small-hf": (
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "fd2c22027eaf20374204f14099b8341e1925ad39",
    ),
}

QWEN_MODEL = "qwen3-vl:4b-instruct"
QWEN_DIGEST_PREFIX = "84f7ed8042fa"


def download_hf_models(models_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is missing; install requirements-cu128.txt first"
        ) from exc

    models_dir.mkdir(parents=True, exist_ok=True)
    for local_name, (repo_id, revision) in HF_MODELS.items():
        target = models_dir / local_name
        print(f"downloading {repo_id}@{revision} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
        )


def download_qwen(model: str) -> None:
    command = ["ollama", "pull", model]
    print("running:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "ollama is not installed; install it before pulling the local VLM"
        ) from exc
    if model == QWEN_MODEL:
        listing = subprocess.check_output(["ollama", "list"], text=True)
        matching = [
            line for line in listing.splitlines() if line.startswith(QWEN_MODEL)
        ]
        if not matching or QWEN_DIGEST_PREFIX not in matching[0]:
            raise SystemExit(
                f"{QWEN_MODEL} digest does not match pinned prefix "
                f"{QWEN_DIGEST_PREFIX}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path.home() / ".local" / "share" / "vision_avoid" / "models",
    )
    parser.add_argument("--qwen-model", default=QWEN_MODEL)
    parser.add_argument(
        "--skip-ollama", action="store_true", help="download only Transformers models"
    )
    args = parser.parse_args()
    download_hf_models(args.models_dir)
    if not args.skip_ollama:
        download_qwen(args.qwen_model)


if __name__ == "__main__":
    main()
