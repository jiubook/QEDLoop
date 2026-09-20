# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Descriptive statistics with one injected defect."""

from __future__ import annotations

from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean.

    Documented contract: raise ``ValueError`` for an empty sequence, because a
    mean of nothing is undefined and a silent zero is indistinguishable from a
    genuine zero mean.
    """
    if not values:
        return 0.0  # BUG: stats/mean_zero -- empty input must raise, not return 0.0
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median() of empty sequence")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def spread(values: Sequence[float]) -> float:
    """Peak-to-peak range; raises for empty input like :func:`mean` should."""
    if not values:
        raise ValueError("spread() of empty sequence")
    return float(max(values) - min(values))
