"""Configuration loading with environment expansion and CLI overrides."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


class ConfigError(ValueError):
    """Raised when a project configuration is malformed."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise ConfigError("PyYAML is required to load .yaml files") from exc
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value or {}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _parse_scalar(value: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(value)
    except ImportError:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``key.subkey=value`` overrides to a copied configuration."""

    result = deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}'; expected key=value")
        dotted_key, raw_value = item.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ConfigError(f"Invalid override key in '{item}'")
        cursor = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot override nested key below '{key}'")
            cursor = child
        cursor[keys[-1]] = _parse_scalar(raw_value)
    return result


def load_config(
    path: str | Path,
    *,
    overrides: Iterable[str] = (),
    required_sections: Iterable[str] = (),
) -> dict[str, Any]:
    """Load JSON/YAML config, expand environment variables, and validate sections."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration does not exist: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        config = _read_yaml(config_path)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    else:
        raise ConfigError(f"Unsupported configuration format: {suffix}")
    if not isinstance(config, dict):
        raise ConfigError("Top-level configuration value must be a mapping")
    config = _expand(apply_overrides(config, overrides))
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing required configuration sections: {missing}")
    config["_config_path"] = str(config_path.resolve())
    return config

