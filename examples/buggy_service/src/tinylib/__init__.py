# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""tinylib -- a deliberately imperfect demo package.

This fixture exists so the loop can be exercised end to end without a cloud
model.  Each module declares its own defect with a marker comment naming the defect;
:mod:`qedloop.core` reads those markers as *asserted* defects, and Phase 5
re-checks them after the candidate patch to decide whether the defect is gone.

Injected defects:

======================  ==================================================
marker                  what is wrong
======================  ==================================================
``stats/mean_zero``     ``mean([])`` silently returns ``0.0``
``mathx/clamp_inverted``  ``clamp()`` compares the wrong way round
``text/title_case_off_by_one``  ``title_case`` mishandles hyphenated words
======================  ==================================================
"""

__version__ = "0.1.0"
