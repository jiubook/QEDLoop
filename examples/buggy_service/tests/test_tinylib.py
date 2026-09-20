# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Baseline suite for the demo fixture.

Two of these tests fail against the shipped fixture: they are the measurement
Phase 5 uses to prove a candidate patch actually fixed something.  A run whose
patch does not turn this suite green cannot converge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinylib.mathx import clamp, lerp  # noqa: E402
from tinylib.stats import mean, median, spread  # noqa: E402
from tinylib.text import slugify, title_case  # noqa: E402


# --- contract expected to hold today --------------------------------------- #


def test_mean_of_values():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_odd_and_even():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5


def test_spread_raises_on_empty():
    with pytest.raises(ValueError):
        spread([])


def test_slugify_collapses_separators():
    assert slugify("Hello World--Again") == "hello-world-again"


def test_lerp_endpoints():
    assert lerp(0.0, 10.0, 0.0) == 0.0
    assert lerp(0.0, 10.0, 1.0) == 10.0


# --- contract that the fixture currently violates -------------------------- #


def test_mean_of_empty_sequence_raises():
    """``mean([])`` must raise: a silent 0.0 is indistinguishable from a real zero."""
    with pytest.raises(ValueError):
        mean([])


def test_clamp_keeps_in_range_values():
    """``clamp`` must not move a value that is already inside the range."""
    assert clamp(5, 0, 10) == 5
    assert clamp(-3, 0, 10) == 0
    assert clamp(42, 0, 10) == 10


def test_title_case_capitalises_hyphenated_words():
    assert title_case("hello-world") == "Hello-World"
