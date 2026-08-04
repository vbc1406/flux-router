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
cost, model_id, tenant_id, run_id, task_type, step_type, a timestamp, and the
routing telemetry listed on UsageRecord (latency, savings, complexity score,
cache hit, routing priority). None of it derives from prompt content. See
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
    """One recorded spend event. Costs and metadata only — never prompt/response text.

    usage_source: "provider" when cost_usd/input_tokens/output_tokens were
    computed from the provider's own reported usage; "estimated" when the
    provider didn't report usage (or the call failed) and cost_usd falls
    back to the pre-dispatch estimate. input_tokens/output_tokens are only
    ever populated ("provider") or None ("estimated") — never a guess.
    """

    tenant_id: str | None
    run_id: str | None
    task_type: str
    step_type: str
    model_id: str
    cost_usd: float
    timestamp: float  # unix epoch seconds
    usage_source: str = "estimated"
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Routing telemetry, added for the local operator dashboard. All optional:
    # the library dispatch path doesn't measure wall-clock latency, and older
    # rows written before these columns existed read back as None/False. Still
    # costs and metadata only — nothing here derives from prompt content.
    latency_ms: float | None = None  # provider call, wall clock
    decision_latency_ms: float | None = None  # routing decision only
    estimated_savings_usd: float | None = None  # vs the most-expensive-model baseline
    complexity_score: float | None = None
    cache_hit: bool = False
    routing_priority: str | None = None
    fallback_used: bool = False


# Column order for the usage table, matching UsageRecord's field order exactly.
# Single source of truth for the INSERT and SELECT below, so the positional
# UsageRecord(*row) construction in query() can't drift out of sync with the
# writer when a column is added.
_USAGE_COLUMNS: tuple[str, ...] = (
    "tenant_id",
    "run_id",
    "task_type",
    "step_type",
    "model_id",
    "cost_usd",
    "timestamp",
    "usage_source",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "decision_latency_ms",
    "estimated_savings_usd",
    "complexity_score",
    "cache_hit",
    "routing_priority",
    "fallback_used",
)

# Columns SQLite stores as INTEGER 0/1 but UsageRecord declares as bool.
_BOOL_COLUMNS: frozenset[str] = frozenset({"cache_hit", "fallback_used"})


def _row_to_record(row: tuple) -> UsageRecord:
    """Build a UsageRecord from a row selected in _USAGE_COLUMNS order.

    SQLite has no boolean type, so cache_hit/fallback_used come back as 0/1
    ints; convert them so callers (and GET /v1/usage's JSON) see real bools.
    """
    values = [
        bool(row[i]) if col in _BOOL_COLUMNS else row[i]
        for i, col in enumerate(_USAGE_COLUMNS)
    ]
    return UsageRecord(*values)


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

    # Columns added after the original schema — see _migrate_schema(). Kept
    # as a single source of truth for both the migration check and the
    # INSERT statement below.
    _MIGRATED_COLUMNS: tuple[tuple[str, str], ...] = (
        ("usage_source", "TEXT NOT NULL DEFAULT 'estimated'"),
        ("input_tokens", "INTEGER"),
        ("output_tokens", "INTEGER"),
        # Routing telemetry for the dashboard. Nullable / defaulted so an
        # existing database opens and migrates in place; pre-existing rows
        # simply read back with no latency and no savings attributed.
        ("latency_ms", "REAL"),
        ("decision_latency_ms", "REAL"),
        ("estimated_savings_usd", "REAL"),
        ("complexity_score", "REAL"),
        ("cache_hit", "INTEGER NOT NULL DEFAULT 0"),
        ("routing_priority", "TEXT"),
        ("fallback_used", "INTEGER NOT NULL DEFAULT 0"),
    )

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
                    timestamp REAL NOT NULL,
                    usage_source TEXT NOT NULL DEFAULT 'estimated',
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    latency_ms REAL,
                    decision_latency_ms REAL,
                    estimated_savings_usd REAL,
                    complexity_score REAL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    routing_priority TEXT,
                    fallback_used INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._migrate_schema()
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_tenant ON usage(tenant_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_run ON usage(run_id)")
            # Every dashboard aggregate filters on a time window, and the
            # summary queries scan the whole table without it.
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp)")
            self._conn.commit()

        self._queue: queue.Queue[UsageRecord] = queue.Queue(
            maxsize=ATTRIBUTION_WRITE_QUEUE_MAX_SIZE
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="flux-usage-writer", daemon=True
        )
        self._writer_thread.start()

    def _migrate_schema(self) -> None:
        """Add usage_source/input_tokens/output_tokens to a pre-existing
        on-disk usage table that predates them. Idempotent and safe to run
        on every startup: CREATE TABLE IF NOT EXISTS above already gives a
        brand-new database these columns directly, so PRAGMA table_info()
        here only ever finds work to do against an OLDER file. Must be
        called with self._lock held. Old rows read back with
        usage_source="estimated" and NULL token counts — see MIGRATIONS.md.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(usage)").fetchall()}
        for column, ddl_type in self._MIGRATED_COLUMNS:
            if column not in existing:
                self._conn.execute(f"ALTER TABLE usage ADD COLUMN {column} {ddl_type}")

    def _writer_loop(self) -> None:
        while True:
            rec = self._queue.get()
            try:
                with self._lock:
                    # Column list and placeholders are both derived from the
                    # fixed _USAGE_COLUMNS tuple — no caller data reaches the
                    # SQL text, every value goes through a ? placeholder.
                    self._conn.execute(
                        f"INSERT INTO usage ({', '.join(_USAGE_COLUMNS)}) "  # noqa: S608 # nosec B608
                        f"VALUES ({', '.join('?' * len(_USAGE_COLUMNS))})",
                        tuple(getattr(rec, col) for col in _USAGE_COLUMNS),
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
            # `where` is assembled only from the fixed clause strings above —
            # tenant_id/run_id values always travel through `params` as `?`
            # placeholders, never interpolated directly. Not injectable.
            cur = self._conn.execute(
                f"SELECT {', '.join(_USAGE_COLUMNS)} "  # noqa: S608 # nosec B608
                f"FROM usage {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            )
            rows = cur.fetchall()
        return [_row_to_record(row) for row in rows]

    def count(self, *, tenant_id: str | None = None, run_id: str | None = None) -> int:
        self._flush()
        where, params = self._where(tenant_id, run_id)
        with self._lock:
            # Same fixed-clause `where` as query() above — not injectable.
            cur = self._conn.execute(f"SELECT COUNT(*) FROM usage {where}", params)  # noqa: S608 # nosec B608
            return cur.fetchone()[0]

    # ── Aggregates for the local dashboard (GET /v1/stats/*) ─────────────────
    #
    # These are deliberately NOT part of the UsageStore protocol: a custom
    # store (the Postgres extension point above) only has to implement
    # record/query/count, and CostAttribution reports the stats surface as
    # unavailable rather than breaking when they're absent. Every one of them
    # aggregates in SQL rather than pulling rows into Python, so a long
    # history stays cheap to render.

    def _stats_where(self, tenant_id: str | None, since: float | None) -> tuple[str, list]:
        clauses, params = [], []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def _percentile(self, where: str, params: list, fraction: float) -> float | None:
        """Nearest-rank percentile of latency_ms over the matching rows.

        SQLite has no percentile function, so this is done as a count plus an
        indexed OFFSET seek. Rows with no latency recorded (pre-migration
        history, or the library path which doesn't time calls) are excluded
        rather than counted as zero — a missing measurement is not a fast one.
        """
        latency_where = (
            f"{where} AND latency_ms IS NOT NULL" if where else "WHERE latency_ms IS NOT NULL"
        )
        total = self._conn.execute(
            f"SELECT COUNT(*) FROM usage {latency_where}", params  # noqa: S608 # nosec B608
        ).fetchone()[0]
        if total == 0:
            return None
        offset = min(int(total * fraction), total - 1)
        row = self._conn.execute(
            f"SELECT latency_ms FROM usage {latency_where} "  # noqa: S608 # nosec B608
            f"ORDER BY latency_ms LIMIT 1 OFFSET ?",
            (*params, offset),
        ).fetchone()
        return row[0] if row else None

    def summary(self, *, since: float | None = None, tenant_id: str | None = None) -> dict:
        """Headline numbers for the selected window."""
        self._flush()
        where, params = self._stats_where(tenant_id, since)
        with self._lock:
            # Fixed-clause `where` built above; all values are ? params.
            row = self._conn.execute(
                f"""SELECT COUNT(*),
                           COALESCE(SUM(cost_usd), 0),
                           COALESCE(SUM(estimated_savings_usd), 0),
                           COALESCE(SUM(CASE WHEN usage_source = 'provider'
                                             THEN cost_usd ELSE 0 END), 0),
                           SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END),
                           SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END),
                           AVG(latency_ms),
                           COUNT(DISTINCT model_id),
                           COUNT(DISTINCT run_id)
                    FROM usage {where}""",  # noqa: S608 # nosec B608
                params,
            ).fetchone()
            p50 = self._percentile(where, params, 0.50)
            p95 = self._percentile(where, params, 0.95)

        requests, cost, savings, actual_cost, cache_hits, fallbacks, avg_lat, models, runs = row
        baseline = cost + savings
        return {
            "requests": requests,
            "runs": runs,
            "distinct_models": models,
            "total_cost_usd": cost,
            # What the same traffic would have cost on the registry's most
            # expensive model — the baseline RoutingDecision.estimated_savings
            # is computed against. Savings is a projection, not a measurement.
            "baseline_cost_usd": baseline,
            "estimated_savings_usd": savings,
            "savings_pct": (savings / baseline * 100) if baseline > 0 else 0.0,
            # How much of total_cost_usd came from provider-reported usage
            # rather than our pre-dispatch estimate. Low values mean the
            # headline number is softer than it looks.
            "actual_cost_usd": actual_cost,
            "actual_cost_pct": (actual_cost / cost * 100) if cost > 0 else 0.0,
            "cache_hits": cache_hits or 0,
            "cache_hit_rate": ((cache_hits or 0) / requests * 100) if requests else 0.0,
            "fallbacks": fallbacks or 0,
            "fallback_rate": ((fallbacks or 0) / requests * 100) if requests else 0.0,
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
        }

    def timeseries(
        self,
        *,
        since: float | None = None,
        bucket_seconds: int = 3600,
        tenant_id: str | None = None,
    ) -> list[dict]:
        """Cost and request count per fixed-width time bucket, oldest first.

        Buckets are aligned to epoch multiples of bucket_seconds so they stay
        stable as the window slides. Empty buckets are absent rather than
        zero-filled — the caller knows the width and can fill gaps itself.
        """
        self._flush()
        where, params = self._stats_where(tenant_id, since)
        bucket = max(1, int(bucket_seconds))
        with self._lock:
            # bucket is coerced to int above, never a caller string.
            rows = self._conn.execute(
                f"""SELECT CAST(timestamp / {bucket} AS INTEGER) * {bucket} AS bucket,
                           COUNT(*),
                           COALESCE(SUM(cost_usd), 0),
                           COALESCE(SUM(estimated_savings_usd), 0),
                           AVG(latency_ms)
                    FROM usage {where}
                    GROUP BY bucket ORDER BY bucket ASC""",  # noqa: S608 # nosec B608
                params,
            ).fetchall()
        return [
            {
                "bucket_start": r[0],
                "requests": r[1],
                "cost_usd": r[2],
                "estimated_savings_usd": r[3],
                "avg_latency_ms": r[4],
            }
            for r in rows
        ]

    def by_model(self, *, since: float | None = None, tenant_id: str | None = None) -> list[dict]:
        """Per-model breakdown, most expensive first."""
        self._flush()
        where, params = self._stats_where(tenant_id, since)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT model_id, COUNT(*), COALESCE(SUM(cost_usd), 0),
                           AVG(latency_ms), COALESCE(SUM(estimated_savings_usd), 0),
                           COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0)
                    FROM usage {where}
                    GROUP BY model_id ORDER BY SUM(cost_usd) DESC""",  # noqa: S608 # nosec B608
                params,
            ).fetchall()
        total_requests = sum(r[1] for r in rows) or 1
        return [
            {
                "model_id": r[0],
                "requests": r[1],
                "cost_usd": r[2],
                "avg_latency_ms": r[3],
                "estimated_savings_usd": r[4],
                "input_tokens": r[5],
                "output_tokens": r[6],
                "share_pct": r[1] / total_requests * 100,
            }
            for r in rows
        ]

    def by_task_type(
        self, *, since: float | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        """Per-task-type breakdown, most expensive first."""
        self._flush()
        where, params = self._stats_where(tenant_id, since)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT task_type, COUNT(*), COALESCE(SUM(cost_usd), 0),
                           AVG(complexity_score), AVG(latency_ms)
                    FROM usage {where}
                    GROUP BY task_type ORDER BY SUM(cost_usd) DESC""",  # noqa: S608 # nosec B608
                params,
            ).fetchall()
        total_requests = sum(r[1] for r in rows) or 1
        return [
            {
                "task_type": r[0],
                "requests": r[1],
                "cost_usd": r[2],
                "avg_complexity_score": r[3],
                "avg_latency_ms": r[4],
                "share_pct": r[1] / total_requests * 100,
            }
            for r in rows
        ]


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
        # Subset of _cost_by_label's totals where usage_source=="provider" —
        # a separate counter (flux_actual_cost_usd_total) so operators can
        # see actual-vs-estimated drift instead of just a blended total. The
        # existing flux_cost_usd_total counter's name and semantics (every
        # recorded dollar, actual or estimated) are unchanged.
        self._actual_cost_by_label: dict[tuple[str, str], float] = defaultdict(float)
        self._run_steps_by_tenant: dict[str, int] = defaultdict(int)
        self._budget_exceeded_by_tenant: dict[str, int] = defaultdict(int)
        # Cardinality cap is shared across ALL counters (cost, actual cost,
        # run steps, budget-exceeded) — capping only flux_cost_usd_total
        # would still let a hostile/buggy caller blow up flux_run_steps by
        # minting a fresh tenant_id per request.
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
        usage_source: str = "estimated",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        decision_latency_ms: float | None = None,
        estimated_savings_usd: float | None = None,
        complexity_score: float | None = None,
        cache_hit: bool = False,
        routing_priority: str | None = None,
        fallback_used: bool = False,
    ) -> None:
        """Record one completed dispatch. Never call with prompt/response text.

        usage_source/input_tokens/output_tokens: pass through from
        ProviderResult when the provider reported actual usage; leave at the
        defaults ("estimated", None, None) when it didn't. See UsageRecord.

        The remaining arguments are routing telemetry for the dashboard and
        are all optional — callers that don't measure them (the library
        dispatch path doesn't time provider calls) leave them at their
        defaults and the columns read back as NULL/False. They feed the
        /v1/stats aggregates only; the Prometheus counters below are
        deliberately unchanged, so no new label dimensions are introduced.
        """
        self._store.record(
            UsageRecord(
                tenant_id,
                run_id,
                task_type,
                step_type,
                model_id,
                cost_usd,
                time.time(),
                usage_source,
                input_tokens,
                output_tokens,
                latency_ms,
                decision_latency_ms,
                estimated_savings_usd,
                complexity_score,
                cache_hit,
                routing_priority,
                fallback_used,
            )
        )
        with self._lock:
            tenant = self._tenant_for(tenant_id)
            label = (
                (_OVERFLOW_LABEL, _OVERFLOW_LABEL)
                if tenant == _OVERFLOW_LABEL
                else (tenant, model_id)
            )
            self._cost_by_label[label] += cost_usd
            if usage_source == "provider":
                self._actual_cost_by_label[label] += cost_usd
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

    @property
    def stats_available(self) -> bool:
        """Whether the configured store can serve the /v1/stats aggregates.

        SqliteUsageStore can; a custom store implementing only the three
        UsageStore protocol methods can't, and the endpoints report that
        rather than raising. Checked against summary() as the representative
        method — they're added and removed as a set.
        """
        return hasattr(self._store, "summary")

    def stats_summary(self, *, since: float | None = None, tenant_id: str | None = None) -> dict:
        return self._store.summary(since=since, tenant_id=tenant_id)  # type: ignore[attr-defined]

    def stats_timeseries(
        self,
        *,
        since: float | None = None,
        bucket_seconds: int = 3600,
        tenant_id: str | None = None,
    ) -> list[dict]:
        return self._store.timeseries(  # type: ignore[attr-defined]
            since=since, bucket_seconds=bucket_seconds, tenant_id=tenant_id
        )

    def stats_by_model(
        self, *, since: float | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        return self._store.by_model(since=since, tenant_id=tenant_id)  # type: ignore[attr-defined]

    def stats_by_task_type(
        self, *, since: float | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        return self._store.by_task_type(since=since, tenant_id=tenant_id)  # type: ignore[attr-defined]

    def render_prometheus(self, tenant_filter: str | None = None) -> str:
        """Render current counters in Prometheus text exposition format.

        tenant_filter: when provided, only series for that (already-normalized,
        see _tenant_for) tenant label are rendered. Used by GET /metrics in
        FLUX_SERVER_TOKENS multi-tenant mode so an authenticated caller can
        only ever see their own bound tenant's spend/usage, not every
        tenant's — see server.py::metrics().
        """
        lines = [
            "# HELP flux_cost_usd_total Total recorded cost in USD (actual or "
            "estimated), labelled by tenant and model.",
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
                "# HELP flux_actual_cost_usd_total Total cost in USD computed from "
                "provider-reported usage only (excludes estimated-fallback spend), "
                "labelled by tenant and model.",
                "# TYPE flux_actual_cost_usd_total counter",
            ]
            for (tenant, model), cost in sorted(self._actual_cost_by_label.items()):
                if tenant_filter is not None and tenant != tenant_filter:
                    continue
                lines.append(
                    f'flux_actual_cost_usd_total{{tenant_id="{_escape_label(tenant)}",'
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
