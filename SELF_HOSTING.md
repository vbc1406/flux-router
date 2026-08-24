# Self-hosting Flux

Running `flux serve` gives you an OpenAI-compatible proxy that does two jobs at
once: it routes each request to the cheapest model that meets your quality bar,
and it caps what any single agent run is allowed to spend before it happens,
not after. On top of that you get a SQLite database of what every request
cost and a dashboard over it — all on your own machine. No traffic reaches us.
There is no telemetry, no callback, and no account to create; your provider
keys and your prompts stay on the box.

This document covers the server. For the Python library and a step-by-step
install, see the [README](./README.md); for the security model, see
[SECURITY_ARCHITECTURE.md](./SECURITY_ARCHITECTURE.md).

---

## Install and run

Flux isn't on PyPI, so it installs straight from the repo. Python 3.10, 3.11,
or 3.12.

```bash
git clone https://github.com/vbc1406/flux.git   # 1. get the code
cd flux
pip install -e '.[server]'                      # 2. install
export OPENAI_API_KEY=sk-...                     # 3. set a provider key (+ any others you use)
flux serve                                       # 4. start the server
```

For a slower, step-by-step walkthrough of the same four commands, see the
[README Quickstart](./README.md#quickstart).

`flux serve` prints where everything lives and starts serving:

```
Flux 1.0.0
  data dir:  /home/you/.local/share/flux
  usage db:  /home/you/.local/share/flux/flux.db
  API:       http://127.0.0.1:8000/v1
  dashboard: http://127.0.0.1:8000/dashboard
```

`python -m router serve` is equivalent, and `make serve` runs the same thing
from a clone.

### Flags

Every flag maps to an environment variable, so anything you can pass on the
command line you can also set in a systemd unit or a container.

| Flag | Environment variable | Default |
|---|---|---|
| `--host` | `FLUX_SERVER_HOST` | `127.0.0.1`, or `0.0.0.0` when a token is set |
| `--port` | `FLUX_SERVER_PORT` | `8000` |
| `--data-dir` | `FLUX_DATA_DIR` | `$XDG_DATA_HOME/flux`, i.e. `~/.local/share/flux` |
| `--db` | `FLUX_ATTRIBUTION_DB` | `<data-dir>/flux.db` |
| `--no-dashboard` | `FLUX_DASHBOARD=0` | dashboard on |
| `--workers` | `FLUX_SERVER_WORKERS` | `1` |
| — | `FLUX_ALLOWED_HOSTS` | empty; loopback names only |

A flag always wins over the environment variable; leaving a flag off keeps
whatever you exported.

`FLUX_ALLOWED_HOSTS` is a comma-separated list of extra `Host` header values
the unauthenticated dashboard and stats endpoints will answer to — see
[Who can see it](#who-can-see-it) below. You need it only for a same-host
reverse proxy running without a token.

### Auth: single-token vs. multi-tenant

Two env vars gate the proxy and dashboard; pick one depending on how many
callers share this deployment:

| | `FLUX_SERVER_TOKEN` | `FLUX_SERVER_TOKENS` |
|---|---|---|
| Shape | one bearer token | JSON map of `{token: tenant_id}` or `{token: {"tenant_id":..., "plan":...}}` |
| Use case | single caller / single operator | **multi-tenant** — each caller gets its own token, bound tenant, and (optionally) budget plan |
| `tenant_id` / `plan` source | client-declared (`X-Flux-Tenant-Id` header, unverified) | server-bound to the token — a caller **cannot** self-declare either, so per-tenant daily caps and `/v1/stats/tenants` rows can't be spoofed |

**If you're serving more than one tenant, use `FLUX_SERVER_TOKENS`, not
`FLUX_SERVER_TOKEN`.** The single-token mode has no way to stop one caller
from setting `user`/`X-Flux-Tenant-Id` to someone else's identity and reading
their spend or borrowing their budget plan — it's meant for exactly one
trusted caller (you, or one backend service). Example:

```bash
export FLUX_SERVER_TOKENS='{"tok-acme":{"tenant_id":"acme","plan":"business_plan"},"tok-beta":"beta"}'
flux serve --host 0.0.0.0
```

Setting both is redundant (either one is enough); setting neither restricts
the server to loopback (see [Who can see it](#who-can-see-it)). See
`router/config.py`'s `ServerTokenBinding`/`_parse_server_tokens` for the exact
parsing rules and `TENANT_DAILY_CAP_USD` for the per-tenant cap this identity
feeds.

### Persistence

The server is persistent by default and the library is not — that difference is
deliberate. Importing `router` and constructing a `Flux` never creates a file on
disk (matching `ResponseCache(enabled=False)` and `AdaptiveWeights(state_file=None)`),
so the library stays side-effect-free in someone else's application. `flux serve`
is the opinionated half: it creates the data directory and points the usage
database at a file inside it, because a self-hosted router that forgets
everything on restart is not much of a product.

The database holds costs and metadata only — model IDs, token counts, latency,
timestamps. **No prompt or completion text is ever written to it.** Backing up
your spend history is `cp ~/.local/share/flux/flux.db backup.db`; the schema
migrates itself forward when a newer Flux opens an older file, so you can keep
the same database across upgrades.

---

## The dashboard

`http://127.0.0.1:8000/dashboard`. Plain HTML, CSS, and JavaScript — no build
step, no Node, nothing fetched from a CDN at runtime. It ships as package data
inside the wheel and is served by the same process as the API.

It shows, for a selected window (1h / 24h / 7d / 30d / all):

- **Headline tiles** — requests, total cost, estimated savings against an
  always-premium baseline, and latency.
- **Spend over time** — cost and request count per bucket, as a hand-drawn SVG
  chart with no charting library behind it.
- **Models** — per-model requests, cost, average latency, tokens, and share of
  traffic, most expensive first.
- **Task types** — the same split by classified task, with average complexity.
- **Model registry** — every model Flux can route to, with live pricing,
  context windows, capabilities, and current RPM load.
- **Configuration** — the deployment's effective settings, read-only.

### Who can see it

The dashboard shows **every tenant's** spend and this deployment's
configuration to whoever loads the page. That is right for the single-operator
case it is built for, and wrong the moment the port is reachable from
elsewhere. The same rule covers the `/v1/stats/*` endpoints behind it, plus
`/v1/usage` and `/metrics` — gating the page alone would achieve nothing, since
it is only a renderer for those. So:

- **Loopback, no token** — served. This is the normal local case.
- **Non-loopback bind, no token** — refused, and the reason is logged. The
  proxy and its API keep working; only the dashboard is withheld.
- **Any bind, token set** — served, and every `/v1/stats/*` call behind it
  requires the token. Paste the token into the dashboard's prompt on first
  load; it is kept in `localStorage` and sent as a bearer header.
- **`FLUX_DASHBOARD=0`** — never served.

The check runs twice: once when deciding whether to mount at all, and again on
every request against the address it actually arrived from. The second check
matters because the first reads the *configured* bind address, which is only
the truth when the bind came from the environment — start the server with
`uvicorn router.server:app --host 0.0.0.0` and the configured value still says
loopback.

"Loopback" means the peer address *and* the `Host` header. A page you visit
elsewhere can point its own hostname at `127.0.0.1` and make your browser
fetch these endpoints for it — the connection is loopback, but the request was
not meant for this machine. The `Host` header is what tells the two apart.

A reverse proxy on the same host still reaches the dashboard, since its peer
address is loopback. If it forwards its own hostname rather than a loopback
one, list it in `FLUX_ALLOWED_HOSTS` (comma-separated). That deployment has
taken responsibility for its own edge and should really set a token, which
turns this whole gate off.

---

## The stats API

The dashboard is a client of these; nothing is available to it that isn't
available to you. All accept `?window=` of `1h`, `24h`, `7d`, `30d`, or `all`
(default `24h`), and an unknown window is a `400` rather than a silent default.

In `FLUX_SERVER_TOKENS` multi-tenant mode, a bearer token's bound tenant is the
only data these return. Without tokens configured there is no verified tenant
identity, so they span every tenant — which is what the single-operator
dashboard wants, and why, with no auth configured, they are served to loopback
callers only. A remote read of any of them (including a Prometheus scrape of
`/metrics`) needs `FLUX_SERVER_TOKEN`.

| Endpoint | Returns |
|---|---|
| `GET /v1/stats/summary` | `requests`, `runs`, `distinct_models`, `total_cost_usd`, `baseline_cost_usd`, `estimated_savings_usd`, `savings_pct`, `actual_cost_usd`, `actual_cost_pct`, `cache_hits`, `cache_hit_rate`, `fallbacks`, `fallback_rate`, `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms` |
| `GET /v1/stats/timeseries` | `data[]` of `bucket_start`, `requests`, `cost_usd`, `estimated_savings_usd`, `avg_latency_ms`. Bucket width defaults per window; `?bucket_seconds=` overrides and is clamped to 60–86400 |
| `GET /v1/stats/models` | `data[]` of `model_id`, `requests`, `cost_usd`, `avg_latency_ms`, `estimated_savings_usd`, `input_tokens`, `output_tokens`, `share_pct` |
| `GET /v1/stats/tasks` | `data[]` of `task_type`, `requests`, `cost_usd`, `avg_complexity_score`, `avg_latency_ms`, `share_pct` |
| `GET /v1/stats/tenants` | `data[]` of `tenant_id`, `requests`, `cost_usd`, `estimated_savings_usd`, `avg_latency_ms`, `runs`, `distinct_models`, `share_pct`. Untagged traffic groups under a null `tenant_id` so rows always sum to the headline total |
| `GET /v1/stats/registry` | Every routable model with pricing, context window, capabilities, and current load. Static catalog fields are unscoped (same content as models.json); `current_load_rpm` alone is process-global and comes back `null` for a `FLUX_SERVER_TOKENS`-bound caller so one tenant can't read load another tenant generated |
| `GET /v1/stats/config` | `server`, `providers`, `budgets`, `run_limits`, `rate_limit` |

Two things worth knowing when you read the numbers:

- **`estimated_savings_usd` is a projection, not a measurement.** It is the
  difference between what you spent and what the same traffic would have cost
  on the registry's most expensive model. Nobody ran that counterfactual.
- **`actual_cost_pct` is the honesty column.** It is the share of
  `total_cost_usd` that came from provider-reported usage rather than our
  pre-dispatch estimate. A low value means the headline number is softer than
  it looks.

`GET /v1/stats/config` never echoes a secret: no bearer token, no provider API
key, no Redis URL. Auth is reported as a mode name and providers as a
configured yes/no. There is a test asserting no configured secret's value
appears in the response body.

---

## Docker

```bash
FLUX_SERVER_TOKEN=$(openssl rand -hex 32) docker compose up -d
```

The image runs `flux serve`, and `docker-compose.yml` mounts a named volume at
`/data` so the usage database survives `docker compose down` and image
rebuilds. Provider keys are read from your shell or a `.env` file beside the
compose file — they are never baked into the image.

**A token is required to use the dashboard under Docker.** A container never
has a loopback client: the proxy binds `0.0.0.0` inside the container and
requests arrive from the docker bridge, so the unauthenticated localhost
allowance never applies — even when the port is published to `127.0.0.1`.
Without a token you get the API alone.

`make docker-down` stops the stack while keeping the volume. Use
`docker compose down -v` to delete the spend history too.

---

## Run-scoped budget enforcement is per-run-id, not automatic

Run budgets (`max_steps`, `max_cost_usd`, `max_tokens` — see the README's
"Run-scoped budgets" section) only accumulate within one `X-Flux-Run-Id`. A
caller that generates a fresh ID (or sends no header at all) on every
request never crosses any cumulative threshold, no matter how many requests
it makes — each one becomes its own harmless single-step run. That's
intentional for the default wrapper use case (a `base_url` swap with no
headers set at all), but it means the permissive default is not itself an
enforcement guarantee for an actual multi-step agent loop.

If you're running Flux for agent workloads where run-scoped budgets need to
actually hold, set:

```bash
export FLUX_REQUIRE_RUN_ID=true
```

With this set, every `/v1/chat/completions` call must carry a valid,
non-blank `X-Flux-Run-Id` (rejected with `400` before the body is even read
otherwise). Leave it unset for a deployment that's only ever a wrapper
proxy for ad-hoc single calls.

---

## Multiple workers

`--workers N` forks N uvicorn processes. Two caveats the server warns about at
startup rather than silently accepting:

- **Run budgets under-enforce.** Run state lives in a process-local store, so
  each worker only sees the steps it personally handled. Set
  `FLUX_RUN_STORE=redis` (plus `FLUX_REDIS_URL`) to share it.
- **Rate limits multiply.** Buckets are process-local too, so N workers enforce
  roughly N times the configured RPM globally. Divide the setting by the worker
  count, or rate limit at your ingress.

The usage database is fine across workers. It runs in SQLite's default
rollback-journal mode rather than WAL — writes go through a single writer
thread per process, so WAL's concurrent-writer throughput would buy nothing
here, and skipping it keeps the database a single file with no `-wal`/`-shm`
sidecars to lose when copying it.

---

## Upgrading

```bash
git pull
pip install -e '.[server]'
flux serve
```

The database migrates in place. Columns added by a newer version are appended
with defaults, and rows written before they existed read back as `NULL` rather
than as zero — a missing measurement is not a fast one, and the latency
aggregates skip those rows instead of counting them. See
[MIGRATIONS.md](./MIGRATIONS.md).
