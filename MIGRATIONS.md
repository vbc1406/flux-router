# Migrations Guide

## When You Need a Migration

A migration is required any time you change:

- The on-disk format of `router/adaptive_state.json`
- The schema of `router/routing_analytics.jsonl`
- A `config.py` constant that affects stored data
- Any `schemas.py` model that is serialised to disk or sent over the wire

Changes that do NOT need a migration:
- Adding a new config constant with a default value
- Adding a new optional field to `RoutingRequest` or `RoutingDecision` (Pydantic handles missing fields gracefully if you provide a default)
- Changing behaviour that is not persisted anywhere

---

## Adaptive State File (`adaptive_state.json`)

### Format Versions

| Version | Schema | When introduced |
|---------|--------|-----------------|
| v0 | Flat dict `{ "model:task": { ... } }` | Original |
| v1 | `{ "global": {...}, "customers": {...} }` | Per-customer weights added |
| v2 | `{ "version": 2, "global": {...}, "customers": {...} }` | Version marker + signal_stats in snapshots |

The current version is **v2**, defined by `_FORMAT_VERSION = 2` in `adaptive_weights.py`.

### How Auto-Migration Works

`AdaptiveWeights._load()` detects the format version on startup:

- v0 (flat dict) → loaded as global state, `_dirty` set to `_WRITE_INTERVAL` so the file is rewritten to v2 on the next flush.
- v1 (no `version` key) → loaded normally, same immediate flush to v2.
- v2 → loaded cleanly, no flush needed.

You do not need to run a migration script manually — it happens automatically on the first startup after an upgrade.

### How to Bump the Format Version

If you need to change the on-disk schema:

1. Increment `_FORMAT_VERSION` in `adaptive_weights.py`.
2. Add a new migration branch in `_load()` for the old version number.
3. Update the table above.
4. Add a test in `test_adaptive_guardrails.py` (follow the `TestDataMigration` pattern).

**Template:**
```python
# In _load(), inside the "global" in data branch:
file_version = data.get("version", 1)
if file_version == 2:
    # Migrate v2 → v3: add new_field with a default
    for key, entry in data["global"].items():
        entry.setdefault("new_field", 0.0)
    self._dirty = _WRITE_INTERVAL
    log.info("adaptive_weights_migrated", from_version=2, to_version=_FORMAT_VERSION)
```

### Rollback Process

If a migration causes problems:

1. Stop the service.
2. Restore `adaptive_state.json` from backup.
3. Revert the code change.
4. Restart. The old format will load correctly.

The adaptive state file is **not critical** — if lost entirely, the system falls back to static quality ratings and rebuilds learned scores over time. Do not let fear of losing it block a deployment.

---

## Analytics Log (`routing_analytics.jsonl`)

The JSONL log is append-only and schema-flexible (each line is a JSON object). Adding new fields to `RoutingAnalytics._decision_to_dict()` is backward-compatible — old entries simply won't have the new field.

**To add a new analytics field:**

1. Add it in `_decision_to_dict()` in `analytics.py`.
2. Add a default/fallback in any query method that reads it (use `.get("new_field", default)`).
3. No migration needed for existing log entries.

**To remove a field:**

1. Remove it from `_decision_to_dict()`.
2. Update any query methods to handle its absence.
3. Old log entries will still have it; queries must handle both.

---

## Config Migrations

When you change a config constant value (not just its name):

1. Check if the constant affects stored state (adaptive weights, analytics, budget ledger).
2. If yes, consider whether existing stored values need recomputing. Usually they don't — the new value takes effect on the next signal or request.
3. Document the change in the constant's comment with a note like:
   ```python
   # Changed from 10 → 20 on 2026-04-28: raised to require more data before overriding static ratings.
   ```

When you rename a config constant:

1. Keep the old name as an alias for one release cycle to avoid import errors across services:
   ```python
   NEW_NAME: int = 20
   OLD_NAME = NEW_NAME  # deprecated alias; remove after 2026-06-01
   ```
2. Search all files for the old name: `grep -r "OLD_NAME" router/`
3. Update all uses.
4. Remove the alias in the next release.

---

## Provider Schema Changes

When a provider changes its API format (new required field, renamed parameter):

1. Update the relevant `_call_<provider>()` function in `provider_caller.py`.
2. Check `errors.py` — new error codes may need a new exception type.
3. Test with a real API call or a mocked response matching the new format.
4. Update the provider's model entries in `models.json` if capabilities or limits changed.

---

## Attribution Usage Database (`router/attribution.py::SqliteUsageStore`)

The `usage` table (SQLite file at `config.ATTRIBUTION_DB_PATH`, or `:memory:`
by default) records cost/metadata only — never prompt or completion text
(see SECURITY_ARCHITECTURE.md). It gained three columns when actual
provider-reported usage recording was added:

| Column | Type | Added | Meaning |
|---|---|---|---|
| `usage_source` | `TEXT NOT NULL DEFAULT 'estimated'` | actual-usage recording | `"provider"` when `cost_usd`/tokens came from the provider's own reported usage; `"estimated"` when it fell back to the pre-dispatch estimate |
| `input_tokens` | `INTEGER` (nullable) | actual-usage recording | Provider-reported input tokens, or `NULL` when `usage_source="estimated"` |
| `output_tokens` | `INTEGER` (nullable) | actual-usage recording | Provider-reported output tokens, or `NULL` when `usage_source="estimated"` |

### How Auto-Migration Works

`SqliteUsageStore.__init__()` always runs `CREATE TABLE IF NOT EXISTS usage`
with the full current schema (so a brand-new database gets these columns
directly), then calls `_migrate_schema()`, which runs `PRAGMA table_info(usage)`
and issues `ALTER TABLE usage ADD COLUMN ...` for any of the three columns
missing from an existing on-disk file predating them. This runs on every
startup and is a no-op once the file is current — safe to leave in place
indefinitely rather than gating it behind a version check.

Because `usage_source` has a `NOT NULL DEFAULT 'estimated'`, SQLite backfills
every pre-existing row with that value automatically as part of the
`ALTER TABLE`. `input_tokens`/`output_tokens` have no default, so old rows
read back as `NULL` (surfaced as `None` in `UsageRecord`, `null` in
`GET /v1/usage` JSON) — this is expected and requires no cleanup: a row from
before this migration genuinely never had actual usage recorded.

No manual migration script or downtime is required. `GET /v1/usage` and
`GET /metrics` (`flux_actual_cost_usd_total`, additive alongside the
unchanged `flux_cost_usd_total`) both handle a mix of pre- and
post-migration rows transparently.

### How to Add Another Column to This Table

1. Add the column to the `CREATE TABLE IF NOT EXISTS usage (...)` DDL in
   `SqliteUsageStore.__init__` (so new databases get it directly).
2. Add `(column_name, "SQL_TYPE ...")` to `SqliteUsageStore._MIGRATED_COLUMNS`
   (so existing on-disk databases get it via `ALTER TABLE`).
3. Add the field to `UsageRecord`, with a default so old callers that
   construct it positionally/with fewer fields keep working.
4. Update the `INSERT INTO usage (...)` column list and `SELECT` in `query()`.
5. Add a test mirroring `TestSqliteUsageStoreMigration` in
   `test_attribution.py` — write a row with the OLD schema to a temp on-disk
   file, reopen it with `SqliteUsageStore`, and assert the new column reads
   back with its default.

---

## Testing Migrations Safely

Before deploying a migration in production:

1. Copy the current `adaptive_state.json` to a temp file.
2. Run the migration code path against the copy:
   ```python
   aw = AdaptiveWeights(state_file="/tmp/test_state.json")
   print(aw._state)   # verify loaded correctly
   aw.flush()         # verify writes new format
   ```
3. Read back the file and confirm the version marker and all data are intact.
4. The `TestDataMigration` test class in `test_adaptive_guardrails.py` covers the standard migration paths — run it: `pytest -v router/tests/test_adaptive_guardrails.py::TestDataMigration`.
