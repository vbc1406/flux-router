"""Server entry point for a freshly-exec'd interpreter — `python -m router._serve`.

Not called directly. `flux serve` (router/cli.py) sets the environment and then
execs this module, because every constant in config.py — and the SqliteUsageStore
default `db_path` derived from it — is read at *import* time, and `import router`
eagerly pulls in router.attribution. By the time any code inside the package can
set FLUX_ATTRIBUTION_DB, the old value is already baked into module state. A new
process is the only way to guarantee the ordering.

Everything this module needs therefore arrives through the environment, and the
banner it prints reflects what the server actually resolved rather than what the
CLI intended.
"""

from __future__ import annotations

import sys


def main() -> int:
    import uvicorn

    from . import server
    from .cli import _version
    from .config import (
        ATTRIBUTION_DB_PATH,
        DATA_DIR,
        SERVER_HOST,
        SERVER_PORT,
        SERVER_WORKERS,
    )

    # 0.0.0.0 is a bind address, not somewhere a browser can be pointed. Compared
    # against, never bound to — the bind address itself comes from config.
    display_host = (
        "127.0.0.1"
        if SERVER_HOST in {"0.0.0.0", "::"}  # noqa: S104 # nosec B104 - comparison, not a bind
        else SERVER_HOST
    )

    print(f"Flux {_version()}")
    print(f"  data dir:  {DATA_DIR}")
    print(f"  usage db:  {ATTRIBUTION_DB_PATH}")
    print(f"  API:       http://{display_host}:{SERVER_PORT}/v1")
    if server.DASHBOARD_MOUNTED:
        print(f"  dashboard: http://{display_host}:{SERVER_PORT}/dashboard")
    sys.stdout.flush()

    # uvicorn can only fork workers from an import string, not a live app object.
    multi = SERVER_WORKERS > 1
    uvicorn.run(
        "router.server:app" if multi else server.app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=SERVER_WORKERS if multi else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
