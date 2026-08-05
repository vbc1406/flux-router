"""cli.py — the `flux` command line entry point.

    flux serve      run the OpenAI-compatible proxy and the local dashboard
    flux version    print the installed version

argparse only, no new dependency — same approach as router/evals/__main__.py.

`serve` is the opinionated, self-hosted counterpart to importing the library:
it creates a data directory and points the usage database at a file inside it,
so a self-hosted instance persists across restarts without the operator
configuring anything. Every setting is still an environment variable
underneath; the flags just set them before the server is imported.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from .paths import DATA_DB_FILENAME, resolve_data_dir

_SERVER_EXTRA_HINT = (
    "`flux serve` requires the 'server' extra. Install with: pip install flux-router[server]"
)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flux-router")
    except PackageNotFoundError:  # running from a source checkout, not installed
        return "unknown (not installed)"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="flux", description="Flux — intelligent LLM routing.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP proxy and local dashboard")
    s.add_argument("--host", default=None,
                   help="bind address (default: 127.0.0.1, "
                        "or 0.0.0.0 when FLUX_SERVER_TOKEN is set)")
    s.add_argument("--port", type=int, default=None, help="port to listen on (default: 8000)")
    s.add_argument("--data-dir", default=None,
                   help="directory for local state (default: $XDG_DATA_HOME/flux)")
    s.add_argument("--db", default=None,
                   help=f"usage database path (default: <data-dir>/{DATA_DB_FILENAME})")
    s.add_argument("--no-dashboard", action="store_true",
                   help="do not serve the operator dashboard at /dashboard")
    s.add_argument("--workers", type=int, default=None,
                   help="number of worker processes (default: 1)")

    sub.add_parser("version", help="print the installed flux-router version")

    return p.parse_args(argv)


def _configure_environment(args: argparse.Namespace) -> tuple[str, str]:
    """Translate serve flags into the environment router.config reads.

    Must run to completion *before* anything imports router.config, which
    snapshots every setting at import time. Returns (data_dir, db_path).
    """
    data_dir = resolve_data_dir(args.data_dir)
    os.environ["FLUX_DATA_DIR"] = data_dir
    os.makedirs(data_dir, exist_ok=True)

    db_path = args.db or os.environ.get("FLUX_ATTRIBUTION_DB")
    if not db_path:
        db_path = os.path.join(data_dir, DATA_DB_FILENAME)
    elif db_path != ":memory:":
        db_path = os.path.abspath(db_path)
    os.environ["FLUX_ATTRIBUTION_DB"] = db_path

    # Flags win over pre-existing env vars; leaving one unset keeps whatever the
    # operator exported, and failing that config.py's own default.
    if args.host is not None:
        os.environ["FLUX_SERVER_HOST"] = args.host
    if args.port is not None:
        os.environ["FLUX_SERVER_PORT"] = str(args.port)
    if args.workers is not None:
        os.environ["FLUX_SERVER_WORKERS"] = str(args.workers)
    if args.no_dashboard:
        os.environ["FLUX_DASHBOARD"] = "0"

    return data_dir, db_path


def _missing_server_extra() -> bool:
    """Whether the optional `server` extra is absent, without importing it."""
    return any(importlib.util.find_spec(mod) is None for mod in ("fastapi", "uvicorn"))


def _serve(args: argparse.Namespace) -> int:
    _configure_environment(args)

    if _missing_server_extra():
        print(_SERVER_EXTRA_HINT, file=sys.stderr)
        return 1

    # Hand off to a fresh interpreter rather than importing the server here.
    # config.py snapshots the environment at import time and `import router`
    # has already happened by now (it is what loaded this module), so nothing
    # set above would reach router.attribution in-process. See router/_serve.py.
    os.execv(  # noqa: S606 # nosec B606 - fixed argv, no shell, no caller-supplied path
        sys.executable, [sys.executable, "-m", "router._serve"]
    )
    return 0  # pragma: no cover - execv does not return


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "version":
        print(_version())
        return 0
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
