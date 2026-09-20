#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Entry point for the self-iterating development loop.

    python run.py run    --target examples/buggy_service --provider mock
    python run.py run    --target ./myrepo --provider deepseek --model deepseek-chat
    python run.py apply  --run runs/<run_id> --dry-run
    python run.py agents
    python run.py check  --target examples/buggy_service
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qedloop.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
