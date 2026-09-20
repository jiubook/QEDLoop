# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""qedloop -- an AI-driven, self-iterating software development loop.

QED -- *quod erat demonstrandum*, "which was to be demonstrated".  The name is
the rule this package enforces: the loop may not claim convergence until the
demonstration exists, and the demonstration has to come from a real test run
rather than from a model's opinion.

Five phases over one state bus, three agents per phase::

    Phase 1 discover -> Phase 2 refine -> Phase 3 review
                                             |  ^ reject
                                             v  |
    Phase 5 qa     <- Phase 4 patch  <--------+

Phase 5 either converges, or feeds the next discovery cycle.

Quick start::

    python run.py run --target examples/buggy_service --provider mock
    python run.py agents
    python -m pytest tests -q
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import Codebase, FileRecord, Issue, Marker, detect_markers
from .graph import END, START, CompiledGraph, StateGraph, StepLimitExceeded
from .llm import LLMProvider, MockProvider, OpenAICompatProvider, make_provider
from .orchestrator import RunConfig, RunResult, apply_run, load_codebase, load_run, run_loop
from .policy import Policy, convergence, review_route, score_cycle
from .state import CHANNELS, initial_state

__all__ = [
    "__version__",
    # graph engine
    "StateGraph", "CompiledGraph", "START", "END", "StepLimitExceeded",
    # state bus
    "CHANNELS", "initial_state",
    # records
    "Codebase", "FileRecord", "Issue", "Marker", "detect_markers",
    # llm
    "LLMProvider", "MockProvider", "OpenAICompatProvider", "make_provider",
    # policy
    "Policy", "score_cycle", "convergence", "review_route",
    # orchestration
    "RunConfig", "RunResult", "run_loop", "apply_run", "load_codebase", "load_run",
]
