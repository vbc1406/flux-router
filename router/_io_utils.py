"""
File: router/_io_utils.py

Purpose:
Internal I/O helpers shared across router modules. Currently provides
`atomic_write_json` for crash-safe state file writes.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_BASE_DIR = Path.home() / ".flux"


def safe_resolve(path_input: str | Path, base: Path | str | None = None) -> Path:
    """Resolve `path_input` to an absolute path under `base` (default `~/.flux/`).

    Raises ValueError if the resolved path escapes `base` (e.g. via
    `../../../etc/passwd` or a symlink). Creates `base` if missing.
    """
    base_path = Path(base) if base else DEFAULT_BASE_DIR
    base_path.mkdir(parents=True, exist_ok=True)
    base_resolved = base_path.resolve()
    candidate = Path(path_input)
    if not candidate.is_absolute():
        candidate = base_resolved / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Path {resolved} is outside allowed base {base_resolved}"
        ) from exc
    return resolved


def atomic_write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write JSON atomically: temp file in the target directory, fsync, then rename.

    On POSIX `os.replace` is atomic, so a reader either sees the previous
    complete file or the new complete file — never a half-written one. Survives
    process death and SIGKILL mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
