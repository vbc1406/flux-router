import json

import router.analytics as analytics_module
from router.analytics import RoutingAnalytics


def test_live_entries_and_index_are_bounded(monkeypatch):
    monkeypatch.setattr(analytics_module, "ANALYTICS_MAX_ENTRIES", 2)
    analytics = RoutingAnalytics(log_path=None)
    analytics._append({"correlation_id": "old", "estimated_cost": 1.0})
    analytics._append({"correlation_id": "keep", "estimated_cost": 2.0})
    analytics._append({"correlation_id": "new", "estimated_cost": 3.0})

    assert [entry["correlation_id"] for entry in analytics.all_entries()] == ["keep", "new"]
    assert set(analytics._index) == {"keep", "new"}
    analytics.update_actual("old", actual_cost=9.0)
    assert all("actual_cost" not in entry for entry in analytics.all_entries())


def test_duplicate_id_eviction_keeps_index_on_newest_entry(monkeypatch):
    monkeypatch.setattr(analytics_module, "ANALYTICS_MAX_ENTRIES", 2)
    analytics = RoutingAnalytics(log_path=None)
    analytics._append({"correlation_id": "same", "version": 1})
    analytics._append({"correlation_id": "same", "version": 2})
    analytics._append({"correlation_id": "other"})
    analytics.update_actual("same", actual_cost=0.5)

    entries = analytics.all_entries()
    assert entries[0]["version"] == 2
    assert entries[0]["actual_cost"] == 0.5


def test_startup_replay_obeys_live_retention_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(analytics_module, "ANALYTICS_MAX_ENTRIES", 2)
    path = tmp_path / "analytics.jsonl"
    path.write_text(
        "".join(json.dumps({"correlation_id": str(i)}) + "\n" for i in range(4)),
        encoding="utf-8",
    )
    analytics = RoutingAnalytics(log_path=path.name, base_dir=tmp_path)
    assert [entry["correlation_id"] for entry in analytics.all_entries()] == ["2", "3"]
    assert set(analytics._index) == {"2", "3"}
