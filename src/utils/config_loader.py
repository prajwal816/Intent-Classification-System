"""
src/utils/config_loader.py
--------------------------
Load and validate YAML configuration files, returning dot-accessible
`ConfigDict` objects so callers can use `cfg.pipeline.sample_rate` syntax.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigDict(dict):
    """
    A dict subclass whose keys are accessible as attributes.

    Example
    -------
    >>> cfg = ConfigDict({"a": {"b": 1}})
    >>> cfg.a.b
    1
    """

    def __init__(self, data: dict):
        super().__init__()
        for k, v in data.items():
            self[k] = ConfigDict(v) if isinstance(v, dict) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"ConfigDict has no key '{key}'") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConfigDict({dict.__repr__(self)})"


def load_config(path: str | Path) -> ConfigDict:
    """
    Load a YAML config file and return a :class:`ConfigDict`.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the ``.yaml`` file.

    Returns
    -------
    ConfigDict

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    return ConfigDict(raw)


def merge_configs(*configs: ConfigDict) -> ConfigDict:
    """
    Deep-merge multiple :class:`ConfigDict` objects left-to-right.
    Later dicts override earlier ones for duplicate keys.
    """
    merged: dict = {}
    for cfg in configs:
        _deep_merge(merged, dict(cfg))
    return ConfigDict(merged)


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
