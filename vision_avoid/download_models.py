from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path.home() / ".local" / "share" / "vision_avoid" / "models",
    )
    args = parser.parse_args()
    download_hf_models(args.models_dir)


if __name__ == "__main__":
    main()
