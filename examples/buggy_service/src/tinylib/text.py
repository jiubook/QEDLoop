# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""String helpers with one injected defect."""

from __future__ import annotations

KEEP = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def title_case(text: str) -> str:
    """Title-case every whitespace-separated word.

    Documented contract: ``title_case("hello-world") == "Hello-World"`` -- a
    hyphen separates two words and both are capitalised.
    """
    words = text.split(" ")
    out = []
    for word in words:
        if not word:
            out.append(word)
            continue
        chars = [c for c in word if c in KEEP]
        # BUG: text/title_case_off_by_one -- the join below re-escapes the
        # hyphen's neighbour and breaks the capitalisation of "world".
        escaped = "".join(c if i > len(chars) - 2 else "\\" + c for i, c in enumerate(chars))
        out.append(escaped[:1].upper() + escaped[1:].lower())
    return " ".join(out)


def slugify(text: str) -> str:
    """Lower-case, hyphen-separated identifier."""
    out = []
    for char in text.strip():
        if char in KEEP:
            out.append(char.lower())
        elif char.isspace() or char in "_-":
            out.append("-")
    collapsed = "".join(out)
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed.strip("-")
