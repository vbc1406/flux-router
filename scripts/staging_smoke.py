"""Keyless checks for a running production-like Flux staging deployment."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def request_json(url: str, token: str | None = None) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            body = json.load(exc)
        except Exception:  # noqa: BLE001 - error body may not be JSON
            body = {}
        return exc.code, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = request_json(f"{base}/health")
    assert status == 200 and health.get("status") == "ok", (status, health)

    status, _ = request_json(f"{base}/v1/stats/config")
    assert status == 401, f"unauthenticated stats returned {status}, expected 401"

    status, config = request_json(f"{base}/v1/stats/config", args.token)
    assert status == 200, (status, config)
    server = config["server"]
    assert server["auth_mode"] == "bound-tokens", server
    assert server["workers"] >= 2, server
    assert server["run_store_backend"] == "redis", server
    assert server["budget_store_backend"] == "redis", server
    assert server["usage_db_persistent"] is True, server

    status, models = request_json(f"{base}/v1/models", args.token)
    assert status == 200 and models.get("data"), (status, models)
    print(
        "Staging keyless checks passed: authenticated multi-worker Redis deployment, "
        f"persistent attribution, {len(models['data'])} routable models"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
