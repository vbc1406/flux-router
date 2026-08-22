import contextlib
import sys

from scripts import validate_model_catalog
from scripts.validate_model_catalog import DOCUMENTED_PROVIDER_IDS, load_models, validate_local


def _fake_urlopen(body: str):
    """Stand in for urllib.request.urlopen as a context manager returning `body`."""

    def opener(*args, **kwargs):
        @contextlib.contextmanager
        def cm():
            class _Resp:
                def read(self):
                    return body.encode("utf-8")

            yield _Resp()

        return cm()

    return opener


def test_catalog_passes_deterministic_validation():
    assert validate_local(load_models()) == []


def test_every_documented_dispatch_override_is_present():
    models = {model["model_id"]: model for model in load_models()}
    for model_id, provider_model_id in DOCUMENTED_PROVIDER_IDS.items():
        assert models[model_id]["provider_model_id"] == provider_model_id


class TestOnlineValidationFailureModes:
    """`--online` runs on a weekly cron against provider websites we do not
    control. An unreachable page must not fail the build; a model id genuinely
    missing from a page we did read must."""

    def _models(self):
        return [
            {
                "model_id": "gpt-oss-20b",
                "provider": "groq",
                "provider_model_id": "openai/gpt-oss-20b",
                "is_available": True,
            }
        ]

    def test_unreachable_source_warns_and_does_not_fail(self, monkeypatch):
        def boom(*args, **kwargs):
            raise TimeoutError("connection timed out")

        monkeypatch.setattr(validate_model_catalog.urllib.request, "urlopen", boom)

        errors, warnings = validate_model_catalog.validate_online(self._models())

        assert errors == []
        assert any("not checked" in w for w in warnings)
        # Every provider page failed, so every provider is reported unchecked.
        assert len(warnings) == len(validate_model_catalog.OFFICIAL_SOURCES)

    def test_id_missing_from_a_readable_page_is_an_error(self, monkeypatch):
        monkeypatch.setattr(
            validate_model_catalog.urllib.request,
            "urlopen",
            _fake_urlopen("a page listing entirely different models"),
        )

        errors, warnings = validate_model_catalog.validate_online(self._models())

        assert warnings == []
        assert any("openai/gpt-oss-20b" in e and "not found" in e for e in errors)

    def test_id_present_on_the_page_passes(self, monkeypatch):
        monkeypatch.setattr(
            validate_model_catalog.urllib.request,
            "urlopen",
            _fake_urlopen("the catalog lists openai/gpt-oss-20b here"),
        )

        errors, warnings = validate_model_catalog.validate_online(self._models())

        assert errors == []
        assert warnings == []

    def test_unreachable_source_exits_zero_end_to_end(self, monkeypatch, capsys):
        def boom(*args, **kwargs):
            raise OSError("TLS handshake failed")

        monkeypatch.setattr(validate_model_catalog.urllib.request, "urlopen", boom)
        monkeypatch.setattr(sys, "argv", ["validate_model_catalog.py", "--online"])

        assert validate_model_catalog.main() == 0
        err = capsys.readouterr().err
        assert "warning:" in err and "not checked" in err
