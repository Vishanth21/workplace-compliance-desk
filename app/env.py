"""Load ``.env`` into the process environment.

Called at the top of the API entry point and the queue worker, before anything
reads configuration. ``python-dotenv`` does not override variables that are
already set, so a real shell export always wins over the file -- which is what
the compose path needs, since ``docker-compose.yml`` passes values through
``environment:`` from the invoking shell.

**Not loaded under pytest.** ``tests/`` imports ``app.main``, which builds the
analyst at import time from ``ANALYST_MODE``. Loading a developer's ``.env``
there would put the suite into ``ANALYST_MODE=real`` and reach for a live
endpoint -- tests would pass or fail depending on whose machine ran them. The
tests set the variables they need with ``monkeypatch``; this module stays out
of their way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The project root, which is where ``.env`` and ``.env.example`` live.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _running_under_pytest() -> bool:
    """True when the importing process is the test suite.

    ``"pytest" in sys.modules`` is checked rather than
    ``PYTEST_CURRENT_TEST``: the latter is only set while a test *executes*,
    but the import of ``app.main`` happens at collection, before any test runs.
    """
    return "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def load_env(path: Path | None = None) -> bool:
    """Load ``.env`` into ``os.environ``. Returns True if a file was loaded.

    Existing environment variables are never overridden. A missing file is not
    an error: the app runs on defaults, and ``docker compose`` supplies its own
    values. Under pytest this is a no-op that reports False.
    """
    if _running_under_pytest():
        return False

    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - declared dependency, belt and braces
        return False

    return load_dotenv(env_path, override=False)