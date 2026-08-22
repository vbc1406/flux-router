"""
File: router/tests/test_cli.py

Purpose:
Tests for router/cli.py — the `flux` command line entry point — and
router/paths.py.

The load-bearing behaviour here is ordering, not parsing: config.py snapshots
every FLUX_* variable at import time, and `import router` has already loaded
router.attribution (baking ATTRIBUTION_DB_PATH into the SqliteUsageStore
default argument) before any CLI code runs. That is why `flux serve` sets the
environment and then execs a fresh interpreter instead of importing the server
in-process, and why cli.py must not import config to resolve the data
directory. Both are asserted below — they are the kind of thing a refactor
silently breaks, leaving a self-hosted instance quietly back on ":memory:".

How to run:
  pytest -v router/tests/test_cli.py
"""

from __future__ import annotations

import os
import sys

import pytest

from router import cli, paths


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, tmp_path):
    """Keep every test off the developer's real ~/.local/share/flux, and keep
    the FLUX_* vars these tests set out of every later test file.

    `_configure_environment` writes to os.environ directly — that is its whole
    job — so monkeypatch has nothing to undo and the values would otherwise
    outlive this module. Snapshot and restore the whole mapping instead.
    """
    saved = dict(os.environ)
    for var in (
        "FLUX_DATA_DIR",
        "FLUX_ATTRIBUTION_DB",
        "FLUX_SERVER_HOST",
        "FLUX_SERVER_PORT",
        "FLUX_SERVER_WORKERS",
        "FLUX_DASHBOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    yield
    os.environ.clear()
    os.environ.update(saved)


def _serve_args(*argv: str):
    return cli._parse_args(["serve", *argv])


class TestArgParsing:
    def test_serve_defaults(self):
        args = _serve_args()
        assert args.command == "serve"
        assert args.host is None and args.port is None
        assert args.data_dir is None and args.db is None
        assert args.workers is None
        assert args.no_dashboard is False

    def test_serve_flags(self, tmp_path):
        args = _serve_args(
            "--host", "0.0.0.0", "--port", "9000", "--data-dir", str(tmp_path),
            "--db", str(tmp_path / "x.db"), "--workers", "4", "--no-dashboard",
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.data_dir == str(tmp_path)
        assert args.db == str(tmp_path / "x.db")
        assert args.workers == 4
        assert args.no_dashboard is True

    def test_version_command(self, capsys):
        assert cli.main(["version"]) == 0
        assert capsys.readouterr().out.strip()

    def test_no_subcommand_is_an_error(self):
        with pytest.raises(SystemExit):
            cli._parse_args([])

    def test_non_integer_port_is_an_error(self):
        with pytest.raises(SystemExit):
            _serve_args("--port", "eight-thousand")


class TestDataDirResolution:
    def test_defaults_to_xdg_data_home(self, tmp_path):
        data_dir, _ = cli._configure_environment(_serve_args())
        assert data_dir == str(tmp_path / "xdg" / "flux")

    def test_flag_beats_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLUX_DATA_DIR", str(tmp_path / "from-env"))
        data_dir, _ = cli._configure_environment(_serve_args("--data-dir", str(tmp_path / "flag")))
        assert data_dir == str(tmp_path / "flag")

    def test_env_used_when_no_flag(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLUX_DATA_DIR", str(tmp_path / "from-env"))
        data_dir, _ = cli._configure_environment(_serve_args())
        assert data_dir == str(tmp_path / "from-env")

    def test_relative_path_is_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir, _ = cli._configure_environment(_serve_args("--data-dir", "reldir"))
        assert os.path.isabs(data_dir)
        assert data_dir == str(tmp_path / "reldir")

    def test_directory_is_created(self, tmp_path):
        target = tmp_path / "nested" / "deeper"
        data_dir, _ = cli._configure_environment(_serve_args("--data-dir", str(target)))
        assert os.path.isdir(data_dir)

    def test_existing_directory_is_reused_not_cleared(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        (target / "flux.db").write_text("prior state")

        cli._configure_environment(_serve_args("--data-dir", str(target)))
        assert (target / "flux.db").read_text() == "prior state"

    def test_config_reuses_the_paths_definitions(self):
        """paths.py owns what config.py and cli.py both need. Asserted as
        identity rather than by comparing resolved values, which would depend
        on the environment config.py happened to be imported under."""
        from router import config

        assert config.default_data_dir is paths.default_data_dir
        assert config.DATA_DB_FILENAME == paths.DATA_DB_FILENAME

    def test_xdg_data_home_is_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-custom"))
        assert paths.default_data_dir() == str(tmp_path / "xdg-custom" / "flux")

    def test_falls_back_to_dot_local_share(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert paths.default_data_dir() == os.path.expanduser("~/.local/share/flux")


class TestDatabasePathResolution:
    def test_defaults_to_a_file_inside_the_data_dir(self, tmp_path):
        data_dir, db_path = cli._configure_environment(
            _serve_args("--data-dir", str(tmp_path / "d"))
        )
        assert db_path == os.path.join(data_dir, paths.DATA_DB_FILENAME)
        assert db_path != ":memory:"

    def test_flag_overrides_the_default(self, tmp_path):
        _, db_path = cli._configure_environment(
            _serve_args("--data-dir", str(tmp_path / "d"), "--db", str(tmp_path / "custom.db"))
        )
        assert db_path == str(tmp_path / "custom.db")

    def test_env_respected_when_no_flag(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLUX_ATTRIBUTION_DB", str(tmp_path / "from-env.db"))
        _, db_path = cli._configure_environment(_serve_args("--data-dir", str(tmp_path / "d")))
        assert db_path == str(tmp_path / "from-env.db")

    def test_memory_is_passed_through_not_absolutized(self, tmp_path):
        _, db_path = cli._configure_environment(
            _serve_args("--data-dir", str(tmp_path / "d"), "--db", ":memory:")
        )
        assert db_path == ":memory:"


class TestEnvironmentHandoff:
    """Everything the child interpreter needs travels through the environment."""

    def test_db_path_is_exported(self, tmp_path):
        _, db_path = cli._configure_environment(_serve_args("--data-dir", str(tmp_path / "d")))
        assert os.environ["FLUX_ATTRIBUTION_DB"] == db_path

    def test_data_dir_is_exported(self, tmp_path):
        data_dir, _ = cli._configure_environment(_serve_args("--data-dir", str(tmp_path / "d")))
        assert os.environ["FLUX_DATA_DIR"] == data_dir

    def test_server_flags_are_exported(self):
        cli._configure_environment(
            _serve_args("--host", "0.0.0.0", "--port", "9000", "--workers", "4")
        )
        assert os.environ["FLUX_SERVER_HOST"] == "0.0.0.0"
        assert os.environ["FLUX_SERVER_PORT"] == "9000"
        assert os.environ["FLUX_SERVER_WORKERS"] == "4"

    def test_no_dashboard_sets_the_off_switch(self):
        cli._configure_environment(_serve_args("--no-dashboard"))
        assert os.environ["FLUX_DASHBOARD"] == "0"

    def test_dashboard_left_alone_by_default(self):
        cli._configure_environment(_serve_args())
        assert "FLUX_DASHBOARD" not in os.environ

    def test_unset_flags_do_not_clobber_operator_env(self, monkeypatch):
        """A flag the operator didn't pass must leave their export in place."""
        monkeypatch.setenv("FLUX_SERVER_HOST", "10.0.0.5")
        monkeypatch.setenv("FLUX_SERVER_PORT", "7000")
        cli._configure_environment(_serve_args())
        assert os.environ["FLUX_SERVER_HOST"] == "10.0.0.5"
        assert os.environ["FLUX_SERVER_PORT"] == "7000"


class TestServeHandoff:
    @pytest.fixture
    def execv_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, argv)))
        return calls

    def test_execs_a_fresh_interpreter_running_the_serve_module(self, execv_calls, tmp_path):
        cli.main(["serve", "--data-dir", str(tmp_path / "d")])

        assert len(execv_calls) == 1
        path, argv = execv_calls[0]
        assert path == sys.executable
        assert argv == [sys.executable, "-m", "router._serve"]

    def test_environment_is_configured_before_the_exec(self, monkeypatch, tmp_path):
        """The ordering the whole design exists for: if FLUX_ATTRIBUTION_DB is
        not already set when the new interpreter starts, it imports config with
        the ":memory:" default and the self-hosted instance silently stops
        persisting."""
        seen = {}
        monkeypatch.setattr(
            os, "execv", lambda path, argv: seen.update(os.environ)  # snapshot at exec time
        )

        cli.main(["serve", "--data-dir", str(tmp_path / "d")])
        assert seen["FLUX_ATTRIBUTION_DB"] == str(tmp_path / "d" / paths.DATA_DB_FILENAME)
        assert seen["FLUX_DATA_DIR"] == str(tmp_path / "d")

    def test_data_dir_exists_before_the_exec(self, monkeypatch, tmp_path):
        """The child opens the SQLite file immediately; the directory has to
        be there already."""
        target = tmp_path / "d"
        existed = {}
        monkeypatch.setattr(
            os, "execv", lambda path, argv: existed.update(there=target.is_dir())
        )

        cli.main(["serve", "--data-dir", str(target)])
        assert existed["there"] is True

    def test_missing_server_extra_reports_the_install_hint(
        self, monkeypatch, capsys, execv_calls, tmp_path
    ):
        monkeypatch.setattr(cli, "_missing_server_extra", lambda: True)

        assert cli.main(["serve", "--data-dir", str(tmp_path / "d")]) == 1
        err = capsys.readouterr().err
        # The hint has to be a command that actually works. Flux is not on PyPI,
        # so `pip install flux-router[server]` — what this used to say — sends
        # the reader to a package that does not exist.
        assert "pip install -e '.[server]'" in err
        assert "flux-router[server]" not in err
        assert execv_calls == []


class TestServeModuleBanner:
    """router/_serve.py runs in the exec'd interpreter and reports what the
    server actually resolved, not what the CLI intended."""

    def test_banner_reports_resolved_config_and_dashboard_url(self, monkeypatch, capsys):
        pytest.importorskip("fastapi")
        uvicorn = pytest.importorskip("uvicorn")
        from router import _serve

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        assert _serve.main() == 0

        out = capsys.readouterr().out
        from router.config import ATTRIBUTION_DB_PATH, DATA_DIR, SERVER_PORT

        assert DATA_DIR in out
        assert ATTRIBUTION_DB_PATH in out
        assert f"http://127.0.0.1:{SERVER_PORT}/v1" in out

    def test_bind_all_is_displayed_as_loopback(self, monkeypatch, capsys):
        """0.0.0.0 is a bind address, not a URL you can open."""
        pytest.importorskip("fastapi")
        uvicorn = pytest.importorskip("uvicorn")
        from router import _serve, config

        monkeypatch.setattr(config, "SERVER_HOST", "0.0.0.0")
        # Not what this test is about — the fail-closed gate on an
        # unauthenticated non-loopback bind is covered separately.
        monkeypatch.setattr(config, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        _serve.main()

        out = capsys.readouterr().out
        assert "http://0.0.0.0" not in out
        assert "http://127.0.0.1" in out

    def test_refuses_unauthenticated_non_loopback_bind_at_startup(self, monkeypatch):
        """The fail-closed check runs when the server actually starts, not as
        an import-time side effect of `import router.config` — otherwise any
        unrelated import sharing this env (tests, other CLI subcommands,
        evals) would crash too."""
        pytest.importorskip("fastapi")
        uvicorn = pytest.importorskip("uvicorn")
        from router import _serve, config

        monkeypatch.setattr(config, "SERVER_HOST", "0.0.0.0")
        monkeypatch.setattr(config, "SERVER_REQUIRE_AUTH", False)
        monkeypatch.setattr(config, "SERVER_ALLOW_UNAUTHENTICATED_REMOTE", False)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

        with pytest.raises(RuntimeError, match="Refusing unauthenticated server bind"):
            _serve.main()

    def test_multiple_workers_uses_an_import_string(self, monkeypatch):
        """uvicorn can only fork workers from an import string, not a live app."""
        pytest.importorskip("fastapi")
        uvicorn = pytest.importorskip("uvicorn")
        from router import _serve, config

        captured = {}
        monkeypatch.setattr(config, "SERVER_WORKERS", 4)
        monkeypatch.setattr(uvicorn, "run", lambda app, **k: captured.update(app=app, **k))
        _serve.main()

        assert captured["app"] == "router.server:app"
        assert captured["workers"] == 4

    def test_single_worker_passes_the_app_object(self, monkeypatch):
        pytest.importorskip("fastapi")
        uvicorn = pytest.importorskip("uvicorn")
        from router import _serve, config, server

        captured = {}
        monkeypatch.setattr(config, "SERVER_WORKERS", 1)
        monkeypatch.setattr(uvicorn, "run", lambda app, **k: captured.update(app=app, **k))
        _serve.main()

        assert captured["app"] is server.app
        assert captured["workers"] is None
