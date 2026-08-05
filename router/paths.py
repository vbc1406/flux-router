"""Filesystem locations for a self-hosted `flux serve`.

Deliberately separate from config.py and importing nothing from it. Every
constant in config.py is read from the environment at *import* time, so
router/cli.py has to finish setting FLUX_ATTRIBUTION_DB before anything
imports config — which rules out asking config where the data directory is.
Keeping the resolution here lets both sides share one definition instead of
duplicating the XDG lookup.
"""

from __future__ import annotations

import os

# Filename of the SQLite usage database inside the data directory.
DATA_DB_FILENAME: str = "flux.db"


def default_data_dir() -> str:
    """Where local state lives when FLUX_DATA_DIR is unset.

    Follows the XDG base-directory spec so it lands somewhere predictable and
    user-owned rather than in whatever directory the server was started from.
    """
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "flux")


def resolve_data_dir(override: str | None = None) -> str:
    """Resolve the data directory: explicit override, then env, then default."""
    return os.path.abspath(override or os.environ.get("FLUX_DATA_DIR") or default_data_dir())
