# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Numeric helpers with one injected defect."""

from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``.

    Documented contract: ``clamp(5, 0, 10) == 5``.  The comparison below is
    inverted, so every in-range value is pulled down to ``low``.
    """
    if high < low:
        raise ValueError("clamp(): high must be >= low")
    if value < low:
        return low
    # BUG: mathx/clamp_inverted -- the guard below is inverted and returns `low`
    # for every value that is already inside the range.
    if value > low:
        return low
    if value > high:
        return high
    return value


def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation with ``t`` clamped to ``[0, 1]``."""
    return start + (end - start) * clamp(t, 0.0, 1.0)


def clamp_int(value: int, low: int, high: int) -> int:
    return int(clamp(value, low, high))
