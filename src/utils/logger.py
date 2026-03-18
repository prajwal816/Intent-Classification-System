"""
src/utils/logger.py
-------------------
Centralized logging configuration for the entire pipeline.
Provides file + console handlers with colorized output via `rich`.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Return (and cache) a named logger with both console and rotating file handlers.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : str
        Logging level string, e.g. ``"INFO"``, ``"DEBUG"``.
    log_dir : str
        Directory for the log file.
    log_file : str
        Log file name inside *log_dir*.
    max_bytes : int
        Maximum size of each log file before rotation.
    backup_count : int
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # avoid duplicate root-logger output

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    if _RICH_AVAILABLE:
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    logger.addHandler(console_handler)

    # ── File handler (rotating) ───────────────────────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)  # always capture DEBUG to file
    logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
