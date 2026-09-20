# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Shared fixtures.

Every test that runs the loop works on a **copy** of the demo fixture, so a test
run can never mutate the checked-in example (and a failing test cannot leave the
repository dirty).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO = ROOT / "examples" / "buggy_service"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def demo_copy(tmp_path: Path) -> Path:
    """A throwaway copy of ``examples/buggy_service`` plus a scratch runs dir."""
    target = tmp_path / "target"
    shutil.copytree(DEMO, target)
    return target


@pytest.fixture(scope="session")
def demo_suite() -> dict:
    """A tiny, self-contained pytest suite used to exercise the sandbox.

    Kept inline (not read from disk) so the sandbox tests do not depend on the
    demo fixture's contents.
    """
    return {
        "mod.py": "def add(a, b):\n    return a + b\n",
        "tests/test_mod.py": (
            "from mod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        ),
    }
