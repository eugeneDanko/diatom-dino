"""Shared JupyterLab runtime for guarded local-GPU workflows.

Long jobs run as child Python processes. This keeps the kernel reusable between
stages, makes CUDA memory reclamation predictable, and guarantees that every
CLI uses the interpreter backing the selected Jupyter kernel.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class NotebookContext:
    project_root: Path
    python: Path
    gpu_index: int | None
    environment: Mapping[str, str]

    def module_command(self, module: str, *arguments: object) -> list[str]:
        return [str(self.python), "-m", module, *(str(value) for value in arguments)]


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root without assuming where JupyterLab was opened."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "PROJECT_CONTEXT.md").is_file():
            return candidate
    raise FileNotFoundError(
        "DiatomDINO root was not found. Start JupyterLab inside the cloned repository."
    )


def bootstrap_notebook(
    *,
    start: str | Path | None = None,
    gpu_index: int | None = 0,
    change_directory: bool = True,
) -> NotebookContext:
    """Create the deterministic environment inherited by every notebook job."""

    if gpu_index is not None and gpu_index < 0:
        raise ValueError("gpu_index must be non-negative or None")
    if gpu_index is not None and "torch" in sys.modules:
        raise RuntimeError(
            "bootstrap_notebook must run before importing torch. Restart the kernel "
            "to change CUDA_VISIBLE_DEVICES safely."
        )
    root = find_project_root(start)
    if change_directory:
        os.chdir(root)

    cache_root = root / ".cache"
    locations = {
        "TORCH_HOME": cache_root / "torch",
        "HF_HOME": cache_root / "huggingface",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root,
        "XDG_CONFIG_HOME": root / ".config",
    }
    for path in locations.values():
        path.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    for name, path in locations.items():
        environment[name] = str(path)
        os.environ[name] = str(path)
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if gpu_index is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    return NotebookContext(
        project_root=root,
        python=Path(sys.executable).resolve(),
        gpu_index=gpu_index,
        environment=environment,
    )


def describe_runtime(context: NotebookContext) -> dict[str, object]:
    report = {
        "project_root": str(context.project_root),
        "kernel_python": str(context.python),
        "python_version": sys.version.split()[0],
        "gpu_index": context.gpu_index,
        "torch_cache": context.environment["TORCH_HOME"],
        "artifacts_root": str(context.project_root / "artifacts"),
        "data_root": str(context.project_root / "data"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def gpu_preflight(
    context: NotebookContext,
    *,
    minimum_vram_gb: float = 0.0,
    require_cuda: bool = True,
) -> dict[str, object]:
    """Validate the CUDA build used by the active Jupyter kernel."""

    import torch

    available = bool(torch.cuda.is_available())
    report: dict[str, object] = {
        "kernel_python": str(context.python),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": available,
        "visible_gpu_count": torch.cuda.device_count() if available else 0,
    }
    if available:
        properties = torch.cuda.get_device_properties(0)
        total_vram_gb = properties.total_memory / 2**30
        report.update({
            "visible_device": 0,
            "physical_gpu_index": context.gpu_index,
            "gpu_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_vram_gb": round(total_vram_gb, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
        if total_vram_gb < minimum_vram_gb:
            raise RuntimeError(
                f"GPU has {total_vram_gb:.1f} GiB VRAM; {minimum_vram_gb:.1f} GiB required"
            )
    elif require_cuda:
        raise RuntimeError(
            "CUDA is unavailable in this Jupyter kernel. Select the DiatomDINO GPU kernel "
            "and install a PyTorch build compatible with the NVIDIA driver."
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def format_command(command: Sequence[object]) -> str:
    values = [str(value) for value in command]
    return subprocess.list2cmdline(values) if os.name == "nt" else shlex.join(values)


def run_guarded(
    context: NotebookContext,
    command: Sequence[object],
    *,
    enabled: bool,
    label: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes] | None:
    """Print a command when disabled and stream it in a child process when enabled."""

    values = [str(value) for value in command]
    print(f"[{label}] {format_command(values)}")
    if not enabled:
        print(f"[{label}] skipped; set the corresponding RUN_* flag to True")
        return None
    process = subprocess.Popen(
        values,
        cwd=context.project_root,
        env=dict(context.environment),
    )
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    completed = subprocess.CompletedProcess(values, return_code)
    if check and return_code:
        raise subprocess.CalledProcessError(return_code, values)
    return completed
