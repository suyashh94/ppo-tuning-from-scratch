#!/usr/bin/env python3
"""Download Qwen 2.5 1.5B base model for torchtune fine-tuning."""

import argparse
import subprocess
import sys
from pathlib import Path


def download_model(output_dir: str, model_id: str = "Qwen/Qwen2.5-1.5B") -> None:
    """Download the Qwen 2.5 1.5B model using torchtune.

    Args:
        output_dir: Directory to save the model files.
        model_id: HuggingFace model ID to download.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "tune",
        "download",
        model_id,
        "--output-dir",
        str(output_path / model_id.split("/")[-1]),
        "--ignore-patterns",
        "None",
    ]

    print(f"Downloading {model_id} to {output_dir}...")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"Model successfully downloaded to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Qwen 2.5 1.5B base model for torchtune fine-tuning."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/base_models",
        help="Directory to save the model (default: /workspace/base_ckpt)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B)",
    )

    args = parser.parse_args()
    download_model(args.output_dir, args.model_id)


if __name__ == "__main__":
    main()
