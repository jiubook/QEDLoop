# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Minimal configuration loading.

``.yml`` support is a ~60-line subset parser (mappings, lists, scalars, comments,
``${ENV}`` expansion) instead of a PyYAML dependency, so the whole framework runs
on a bare CPython.  When PyYAML *is* installed it is used instead, so nothing is
lost on machines that already have it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

ENV_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# tiny YAML subset
# --------------------------------------------------------------------------- #


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_scalar(part) for part in _split_inline(inner)]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_inline(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    current = ""
    quote = ""
    for ch in text:
        if quote:
            current += ch
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            current += ch
        elif ch in "[{":
            depth += 1
            current += ch
        elif ch in "]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def _strip_comment(line: str) -> str:
    quote = ""
    for index, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def load_config(path: str | Path) -> Dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        raise ConfigError("config file not found: %s" % file)
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() in (".json",):
        return json.loads(text)
    try:  # prefer a real YAML parser when the environment has one
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return dict(loaded or {})
    except ImportError:
        return parse_yaml_simple(text)


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path referenced *by a config file*.

    A relative one resolves against the config file's directory rather than the
    working directory, which is what a config-referenced asset normally means:
    ``brief: brief.md`` next to the config that names it.  ``target`` and
    ``out_dir`` are deliberately not treated this way -- they are runtime
    locations chosen by whoever invokes the run.
    """
    file = Path(path)
    if base is not None and not file.is_absolute():
        file = Path(base) / file
    return file


def load_text_file(path: str | Path, *, what: str = "file", base: str | Path | None = None) -> str:
    """Read a UTF-8 text asset referenced by a config file.

    A missing or undecodable file is an error, never an empty constraint set --
    a silently ignored brief is how a run ends up breaking a rule the
    maintainer thought they had stated.
    """
    file = resolve_path(path, base)
    if not file.is_file():
        raise ConfigError("%s file not found: %s" % (what, file))
    try:
        # utf-8-sig: PowerShell happily writes a BOM, and a stray \ufeff would
        # otherwise be sent to the model as part of the first line.
        return file.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError("%s file %s could not be read as UTF-8: %s" % (what, file, exc))


# --------------------------------------------------------------------------- #
# a second, simpler YAML reader (2-space nesting, used when PyYAML is absent)
# --------------------------------------------------------------------------- #


def parse_yaml_simple(text: str) -> Dict[str, Any]:
    """Indentation-based reader supporting nested maps and ``-`` lists."""
    lines = [
        (len(line) - len(line.lstrip(" ")), _strip_comment(line.rstrip()))
        for line in (text or "").splitlines()
    ]
    lines = [(indent, body) for indent, body in lines if body.strip()]
    value, _ = _parse_block(lines, 0, 0 if not lines else lines[0][0])
    if not isinstance(value, dict):
        raise ConfigError("top level of a config file must be a mapping")
    return expand_env(value)


def _parse_block(lines: List[Any], index: int, indent: int):
    if index >= len(lines):
        return {}, index
    if lines[index][1].strip().startswith("- "):
        items: List[Any] = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].strip().startswith("- "):
            body = lines[index][1].strip()[2:].strip()
            index += 1
            if ":" in body and not body.startswith(("{", "[")):
                key, _, rest = body.partition(":")
                row: Dict[str, Any] = {}
                if rest.strip():
                    row[key.strip()] = _scalar(rest)
                else:
                    child, index = _parse_block(lines, index, lines[index][0] if index < len(lines) else indent + 2)
                    row[key.strip()] = child if child != {} else {}
                items.append(row)
            else:
                items.append(_scalar(body))
        return items, index

    mapping: Dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent:
        body = lines[index][1].strip()
        if body.startswith("- "):
            break
        key, _, rest = body.partition(":")
        key = key.strip()
        if rest.strip():
            mapping[key] = _scalar(rest)
            index += 1
            continue
        index += 1
        if index < len(lines) and lines[index][0] > indent:
            child, index = _parse_block(lines, index, lines[index][0])
            mapping[key] = child
        else:
            mapping[key] = {}
    return mapping, index


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_RE.sub(lambda m: os.environ.get(m.group("name"), m.group("default") or ""), value)
    if isinstance(value, Mapping):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# merge helpers
# --------------------------------------------------------------------------- #


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def pick(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """First present, non-None value among dotted paths."""
    for key in keys:
        node: Any = data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return default


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
