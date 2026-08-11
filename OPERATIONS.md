# Flux production operations

This runbook covers the minimum operational procedures for a self-hosted Flux deployment. Adapt commands, paths, retention, and ownership to your platform.

## Pre-deployment record

Record these before every release:

- Flux commit or image digest.
- Configuration values, excluding secrets.
- Database backup location and restore verification time.
- Redis endpoint and persistence policy when using multiple workers.
- Enabled providers and model catalog revision.
- Expected health, latency, fallback, error-rate, and spend thresholds.

Never store provider keys or bearer tokens in the deployment record.

## Production-like staging stack

`docker-compose.staging.yml` exercises the deployment shape that matters in
production: two authenticated Flux workers, Redis-backed shared run budgets
AND Redis-backed shared plan-level (daily/monthly) budgets — two independent
systems (`router/run_budget.py` and `router/budget_tracker.py` respectively;
`FLUX_RUN_STORE=redis` and `FLUX_BUDGET_STORE=redis` must both be set for a
multi-worker deployment) — persistent attribution storage, and a persistent
Redis append-only log.

```bash
cp .env.staging.example .env.staging
# Replace the token placeholder and add only the provider keys being tested.
docker compose --env-file .env.staging -f docker-compose.staging.yml up -d --build
docker compose --env-file .env.staging -f docker-compose.staging.yml ps
python scripts/staging_smoke.py --token 'the-token-key-from-FLUX_SERVER_TOKENS'
```

Keep the published port on host loopback unless a separately authenticated
ingress is ready. Use the tenant-bound bearer token from `FLUX_SERVER_TOKENS`
for the dashboard, stats endpoints, wrapper smoke tests, and agent-run smoke
tests. Stop the stack without deleting its volumes during ordinary testing:

```bash
docker compose --env-file .env.staging -f docker-compose.staging.yml down
```

Use `down -v` only when intentionally discarding staging attribution and Redis
state.

## Backup and restore

Stop Flux or take a SQLite-safe online backup before changing versions:

```bash
sqlite3 "$FLUX_DATA_DIR/flux.db" ".backup '$FLUX_DATA_DIR/flux-backup.db'"
sqlite3 "$FLUX_DATA_DIR/flux-backup.db" "PRAGMA integrity_check;"
```

Copy the verified backup to storage outside the Flux host or container volume. Test restoration periodically in staging. For externally managed databases and Redis, use the platform's snapshot and point-in-time recovery procedures.

## Deployment and rollback

1. Deploy the new immutable image or commit to staging.
2. Run `/health`, an authenticated text completion, streaming completion, tool call, and budgeted multi-step run.
3. Compare routing distribution, provider errors, latency, spend, and fallback rate with the pre-deployment baseline.
4. Roll out gradually when the platform supports canaries.

Rollback triggers include sustained provider errors, unexpected premium-tier inflation, budget under-enforcement, attribution loss, authentication failures, or material latency regression.

To roll back, stop new traffic, restore the previous immutable image/configuration, and restore the database only when a schema or data migration requires it. Do not overwrite a newer database until a recovery copy has been preserved. Re-run the staging smoke sequence before reopening traffic.

## Key and token rotation

- Keep provider keys and `FLUX_SERVER_TOKENS` in a secret manager, never in source control or image layers.
- Create the replacement credential before revoking the old one.
- Update one provider or tenant binding at a time, restart/reload Flux, and verify an authenticated request.
- Revoke the previous credential after verification.
- Rotate immediately after suspected exposure and inspect usage for unexpected tenants, models, or spend.

For multi-tenant deployments, use `FLUX_SERVER_TOKENS`; do not share one legacy token across customers.

## Spend and reliability alerts

Alert on at least:

- Actual spend and estimated-vs-actual cost variance.
- Budget-stop and budget-degradation rate.
- Provider authentication, rate-limit, timeout, and 5xx errors.
- Fallback and circuit-breaker activation rate.
- Premium-tier share and sudden model-distribution changes.
- P50/P95/P99 routing and end-to-end latency.
- Dropped attribution records and persistence failures.
- Redis availability when multiple workers share run budgets and/or plan-level daily/monthly budgets.

Set absolute spend caps at the provider as a final backstop; Flux budgets should not be the only financial control.

## Incident response

1. Contain: disable the affected provider/model or remove external access while preserving evidence.
2. Identify affected tenants, runs, time range, provider calls, spend, and stored attribution data.
3. Rotate exposed credentials and revoke compromised tenant tokens.
4. Restore a known-good image/configuration if a release caused the incident.
5. Reconcile provider usage with Flux attribution and document any missing records.
6. Notify affected operators or customers according to contractual and regulatory obligations.
7. Add a regression test or monitoring rule before closing the incident.

Do not place prompts, credentials, authorization headers, or sensitive provider response bodies in incident tickets or ordinary logs.
