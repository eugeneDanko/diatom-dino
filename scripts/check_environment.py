"""Read-only readiness check for a local NVIDIA GPU workstation."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--minimum-free-gb", type=float, default=0.0,
        help="Fail when the filesystem has less free space; zero only reports it.",
    )
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Do not fail when CUDA is unavailable (useful only for code inspection).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    existing_parent = next((p for p in [data_root, *data_root.parents] if p.exists()), Path.cwd())
    disk = shutil.disk_usage(existing_parent)
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(Path.cwd().resolve()),
        "data_root": str(data_root),
        "disk_free_gb": round(disk.free / 2**30, 2),
        "minimum_free_gb": float(args.minimum_free_gb),
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "torch_cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
    }
    try:
        import torch

        report["torch_installed"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = torch.version.cuda
        if report["cuda_available"]:
            report["gpu_count"] = torch.cuda.device_count()
            report["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gb": round(
                        torch.cuda.get_device_properties(index).total_memory / 2**30, 2
                    ),
                }
                for index in range(torch.cuda.device_count())
            ]
    except (ImportError, OSError) as error:
        report["torch_error"] = str(error)

    failures: list[str] = []
    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or newer is required")
    if disk.free / 2**30 < args.minimum_free_gb:
        failures.append(
            f"Free disk space is below {args.minimum_free_gb:.1f} GiB"
        )
    if not report["torch_installed"]:
        failures.append("PyTorch is not installed")
    elif not report["cuda_available"] and not args.allow_cpu:
        failures.append("CUDA is unavailable to the installed PyTorch build")
    report["ready"] = not failures
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
