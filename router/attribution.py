"""
File: router/attribution.py

Purpose:
Per-run and per-tenant cost attribution (Task 7). Answers "which
customer/workflow is eating my margin" — the number that actually gets
forwarded to a founder or CFO, as opposed to routing accuracy.

Main Classes:
  UsageRecord      — one recorded spend event (cost + metadata, NEVER prompt
                      or completion content)
  SqliteUsageStore — default persistence, stdlib sqlite3, no new dependency
  CostAttribution  — facade: records usage, aggregates, and renders
                      Prometheus text exposition for /metrics

Config Dependencies (all in config.py):
  ATTRIBUTION_DB_PATH                  — SQLite file path (or ":memory:")
  ATTRIBUTION_METRICS_MAX_LABEL_COMBOS — Prometheus label cardinality cap
  ATTRIBUTION_USAGE_PAGE_MAX           — GET /v1/usage page size cap

SECURITY: no prompts or completions are ever stored by this module — only
cost, model_id, tenant_id, run_id, task_type, step_type, and a timestamp. See
SECURITY_ARCHITECTURE.md.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import structlog

from .config import (
    ATTRIBUTION_DB_PATH,
    ATTRIBUTION_METRICS_MAX_LABEL_COMBOS,
    ATTRIBUTION_WRITE_QUEUE_MAX_SIZE,
)

log = structlog.get_logger(__name__)

_OVERFLOW_LABEL = "_overflow_"


def _escape_label(value: str) -> str:
    """Escape a Prometheus label value per the text exposition format: a raw
    backslash, double-quote, or newline in a caller-controlled value (e.g.
    X-Flux-Tenant-Id) would otherwise break the label syntax for every line
    after it, causing scrapers to reject the whole /metrics payload."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class UsageRecord:
    """One recorded spend event. Costs and metadata only — never prompt/response text."""

    tenant_id: str | None
    run_id: str | None
    task_type: str
    step_type: str
    model_id: str
    cost_usd: float
    timestamp: float  # unix epoch seconds


class UsageStore(Protocol):
    def record(self, rec: UsageRecord) -> None: ...
    def query(
        self,
        *,
        tenant_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UsageRecord]: ...
    def count(self, *, tenant_id: str | None = None, run_id: str | None = None) -> int: ...


class SqliteUsageStore:
    """
    Default UsageStore: a single SQLite file (or ":memory:" for tests /
    ephemeral deployments). Fine for one process; NOT shared across
    multiple server instances.

    Writes go through a background thread, not the calling thread: record()
    is called synchronously from async request handlers (the HTTP proxy's
    hot path), and a blocking execute()+commit() there means every request
    pays for a disk fsync serialized behind one lock, on the event loop
    thread. record() instead enqueues and returns immediately; a single
    writer thread drains the queue and does the actual insert+commit.
    query()/count() call _flush() first so reads stay consistent with any
    writes issued before them, at the cost of a (usually tiny) wait on that
    admin/reporting path instead of the hot write path.

    🔧 EXTENSION POINT: implement the UsageStore protocol against Postgres
    for multi-instance deployments — swap the constructor call in
    CostAttribution.__init__, the query surface (record/query/count) stays
    the same.
    """

    def __init__(self, db_path: str = ATTRIBUTION_DB_PATH) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            # WAL mode makes the writer thread's commits cheaper (no-op for
            # ":memory:" databases, which don't support WAL — sqlite silently
            # keeps them in "memory" journal mode instead).
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT,
                    run_id TEXT,
                    task_type TEXT,
                    step_type TEXT,
                    model_id TEXT NOT NULL,
                    cost_usd REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_tenant ON usage(tenant_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_run ON usage(run_id)")
            self._conn.commit()

        self._queue: queue.Queue[UsageRecord] = queue.Queue(
            maxsize=ATTRIBUTION_WRITE_QUEUE_MAX_SIZE
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="flux-usage-writer", daemon=True
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        while True:
            rec = self._queue.get()
            try:
                with self._lock:
                    self._conn.execute(
                        "INSERT INTO usage "
                        "(tenant_id, run_id, task_type, step_type, model_id, cost_usd, timestamp) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            rec.tenant_id,
                            rec.run_id,
                            rec.task_type,
                            rec.step_type,
                            rec.model_id,
                            rec.cost_usd,
                            rec.timestamp,
                        ),
                    )
                    self._conn.commit()
            finally:
                self._queue.task_done()

    def record(self, rec: UsageRecord) -> None:
        try:
            self._queue.put_nowait(rec)
        except queue.Full:
            log.warning(
                "attribution_write_queue_full",
                limit=ATTRIBUTION_WRITE_QUEUE_MAX_SIZE,
                msg="usage-store writer thread is falling behind; dropping this record",
            )

    def _flush(self) -> None:
        """Block until every record enqueued so far has been committed."""
        self._queue.join()

    def _where(self, tenant_id: str | None, run_id: str | None) -> tuple[str, list]:
        clauses, params = [], []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def query(
        self,
        *,
        tenant_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UsageRecord]:
        self._flush()
        where, params = self._where(tenant_id, run_id)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT tenant_id, run_id, task_type, step_type, model_id, cost_usd, timestamp "  # noqa: S608
                f"FROM usage {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            )
            rows = cur.fetchall()
        return [UsageRecord(*row) for row in rows]

    def count(self, *, tenant_id: str | None = None, run_id: str | None = None) -> int:
        self._flush()
        where, params = self._where(tenant_id, run_id)
        with self._lock:
            cur = self._conn.execute(f"SELECT COUNT(*) FROM usage {where}", params)  # noqa: S608
            return cur.fetchone()[0]


class CostAttribution:
    """
    Facade used by routing_engine.py / flux.py to record spend, and by
    router/server.py to serve GET /v1/usage and GET /metrics.

    Prometheus counters are kept in-process (not derived from SQLite on each
    scrape) and cardinality-capped at ATTRIBUTION_METRICS_MAX_LABEL_COMBOS —
    once the cap is hit, further NEW (tenant_id, model_id) pairs are folded
    into an "_overflow_" bucket rather than growing the label set forever, so
    a hostile or buggy caller minting a fresh tenant_id per request can't
    blow up /metrics.
    """

    def __init__(self, store: UsageStore | None = None) -> None:
        self._store: UsageStore = store or SqliteUsageStore()
        self._lock = threading.Lock()
        self._cost_by_label: dict[tuple[str, str], float] = defaultdict(float)
        self._run_steps_by_tenant: dict[str, int] = defaultdict(int)
        self._budget_exceeded_by_tenant: dict[str, int] = defaultdict(int)
        # Cardinality cap is shared across ALL three counters (cost, run
        # steps, budget-exceeded) — capping only flux_cost_usd_total would
        # still let a hostile/buggy caller blow up flux_run_steps by minting
        # a fresh tenant_id per request.
        self._known_tenants: set[str] = set()

    def _tenant_for(self, tenant_id: str | None) -> str:
        tenant = tenant_id or "unknown"
        if (
            tenant in self._known_tenants
            or len(self._known_tenants) < ATTRIBUTION_METRICS_MAX_LABEL_COMBOS
        ):
            self._known_tenants.add(tenant)
            return tenant
        log.info(
            "attribution_label_cardinality_overflow", limit=ATTRIBUTION_METRICS_MAX_LABEL_COMBOS
        )
        return _OVERFLOW_LABEL

    def record(
        self,
        *,
        tenant_id: str | None,
        run_id: str | None,
        task_type: str,
        step_type: str,
        model_id: str,
        cost_usd: float,
    ) -> None:
        """Record one completed dispatch. Never call with prompt/response text."""
        self._store.record(
            UsageRecord(tenant_id, run_id, task_type, step_type, model_id, cost_usd, time.time())
        )
        with self._lock:
            tenant = self._tenant_for(tenant_id)
            label = (
                (_OVERFLOW_LABEL, _OVERFLOW_LABEL)
                if tenant == _OVERFLOW_LABEL
                else (tenant, model_id)
            )
            self._cost_by_label[label] += cost_usd
            self._run_steps_by_tenant[tenant] += 1

    def record_budget_exceeded(self, tenant_id: str | None) -> None:
        with self._lock:
            self._budget_exceeded_by_tenant[self._tenant_for(tenant_id)] += 1

    def usage(
        self,
        *,
        tenant_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[UsageRecord], int]:
        """Return (page, total_count) for GET /v1/usage."""
        records = self._store.query(tenant_id=tenant_id, run_id=run_id, limit=limit, offset=offset)
        total = self._store.count(tenant_id=tenant_id, run_id=run_id)
        return records, total

    def render_prometheus(self, tenant_filter: str | None = None) -> str:
        """Render current counters in Prometheus text exposition format.

        tenant_filter: when provided, only series for that (already-normalized,
        see _tenant_for) tenant label are rendered. Used by GET /metrics in
        FLUX_SERVER_TOKENS multi-tenant mode so an authenticated caller can
        only ever see their own bound tenant's spend/usage, not every
        tenant's — see server.py::metrics().
        """
        lines = [
            "# HELP flux_cost_usd_total Total estimated cost in USD, labelled by tenant and model.",
            "# TYPE flux_cost_usd_total counter",
        ]
        with self._lock:
            for (tenant, model), cost in sorted(self._cost_by_label.items()):
                if tenant_filter is not None and tenant != tenant_filter:
                    continue
                lines.append(
                    f'flux_cost_usd_total{{tenant_id="{_escape_label(tenant)}",'
                    f'model_id="{_escape_label(model)}"}} {cost:.6f}'
                )
            lines += [
                "# HELP flux_run_steps Total routing steps recorded, labelled by tenant.",
                "# TYPE flux_run_steps counter",
            ]
            for tenant, count in sorted(self._run_steps_by_tenant.items()):
                if tenant_filter is not None and tenant != tenant_filter:
                    continue
                lines.append(f'flux_run_steps{{tenant_id="{_escape_label(tenant)}"}} {count}')
            lines += [
                "# HELP flux_budget_exceeded_total Run-budget exceeded events, labelled by tenant.",
                "# TYPE flux_budget_exceeded_total counter",
            ]
            for tenant, count in sorted(self._budget_exceeded_by_tenant.items()):
                if tenant_filter is not None and tenant != tenant_filter:
                    continue
                lines.append(
                    f'flux_budget_exceeded_total{{tenant_id="{_escape_label(tenant)}"}} {count}'
                )
        return "\n".join(lines) + "\n"
