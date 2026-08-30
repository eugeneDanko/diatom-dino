"""Read-only readiness check for a local NVIDIA GPU workstation."""

from __future__ import annotations

import argparse
import importlib.metadata
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
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--minimum-vram-gb", type=float, default=0.0)
    parser.add_argument(
        "--require-jupyter", action="store_true",
        help="Also require JupyterLab and ipykernel in this interpreter.",
    )
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be non-negative")
    if args.minimum_vram_gb < 0:
        parser.error("--minimum-vram-gb must be non-negative")

    data_root = Path(args.data_root).resolve()
    existing_parent = next((p for p in [data_root, *data_root.parents] if p.exists()), Path.cwd())
    disk = shutil.disk_usage(existing_parent)
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
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
        "selected_gpu_index": int(args.gpu_index),
        "jupyterlab_version": None,
        "ipykernel_version": None,
    }
    for package, key in (("jupyterlab", "jupyterlab_version"), ("ipykernel", "ipykernel_version")):
        try:
            report[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
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
            if not 0 <= args.gpu_index < torch.cuda.device_count():
                report["selected_gpu_error"] = (
                    f"GPU index {args.gpu_index} is outside 0..{torch.cuda.device_count()-1}"
                )
            else:
                selected = torch.cuda.get_device_properties(args.gpu_index)
                report["selected_gpu"] = {
                    "index": args.gpu_index,
                    "name": selected.name,
                    "memory_gb": round(selected.total_memory / 2**30, 2),
                    "compute_capability": f"{selected.major}.{selected.minor}",
                }
    except Exception as error:
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
    elif report["cuda_available"]:
        if "selected_gpu_error" in report:
            failures.append(str(report["selected_gpu_error"]))
        elif float(report["selected_gpu"]["memory_gb"]) < args.minimum_vram_gb:
            failures.append(
                f"Selected GPU VRAM is below {args.minimum_vram_gb:.1f} GiB"
            )
    if args.require_jupyter:
        if report["jupyterlab_version"] is None:
            failures.append("JupyterLab is not installed in this interpreter")
        if report["ipykernel_version"] is None:
            failures.append("ipykernel is not installed in this interpreter")
    report["ready"] = not failures
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
