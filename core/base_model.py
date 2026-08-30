"""Common, intentionally small model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def load(self, source: str | Path | None = None) -> "BaseModel": ...

    @abstractmethod
    def to_device(self, device: str | Any) -> "BaseModel": ...

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

