# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""LLM access: one provider interface, three implementations.

* :class:`OpenAICompatProvider` -- any ``/chat/completions`` endpoint
  (OpenAI, DeepSeek, Moonshot, vLLM, Ollama, ...) using only the stdlib.
* :class:`MockProvider` -- a deterministic, offline stand-in.  It is honest
  about what it can do: it only reacts to *declared* defect markers that the
  agent specs ask for, so the whole loop can be exercised without a key.
* :class:`RecordingProvider` -- replay a previous run, or run against a fixture
  transcript in tests.

Design intent: the framework never imports a vendor SDK, so the loop can run in
a locked-down sandbox.  Swap providers with ``--provider`` / ``LLM_PROVIDER``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- #
# messages and replies
# --------------------------------------------------------------------------- #


@dataclass
class Message:
    role: str          # system | user | assistant
    content: str


@dataclass
class LLMReply:
    text: str
    fingerprint: str = ""
    model: str = "?"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    pass


def prompt_fingerprint(provider: str, model: str, messages: Sequence[Message]) -> str:
    h = hashlib.sha256()
    h.update(("%s|%s|" % (provider, model)).encode("utf-8"))
    for m in messages:
        h.update(("%s\x1f%s\x1e" % (m.role, m.content)).encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# base + wrappers
# --------------------------------------------------------------------------- #


class LLMProvider:
    name = "base"
    channel = "base"       # which configured channel produced this instance
    model = "none"
    concurrency = 4
    request_timeout = 90.0   # seconds; local models need a much larger value
    max_tokens: Optional[int] = None   # the channel's output ceiling, if it declared one
    require_key = True       # False for a channel that declares it needs none

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMReply:
        raise NotImplementedError

    def __call__(self, messages: Sequence[Message], **kw: Any) -> LLMReply:
        return self.complete(messages, **kw)


class CachingProvider(LLMProvider):
    """Memoise by prompt fingerprint -- cheap reruns and stable tests."""

    def __init__(self, inner: LLMProvider, max_entries: int = 4096) -> None:
        self.inner = inner
        self.name = "%s+cache" % inner.name
        self.channel = inner.channel
        self.model = inner.model
        self.concurrency = inner.concurrency
        self.request_timeout = getattr(inner, "request_timeout", getattr(inner, "timeout", 90.0))
        # Carry the output ceiling through the wrapper: dropping it here is how
        # a channel that asked for a thinking model's budget silently falls back
        # to the framework's small JSON answer budget.
        self.max_tokens = getattr(inner, "max_tokens", None)
        self.require_key = bool(getattr(inner, "require_key", True))
        self.max_entries = max_entries
        self._store: Dict[str, LLMReply] = {}
        self.hits = 0
        self.misses = 0

    def __getattr__(self, item: str) -> Any:
        """Transparent wrapper: anything not listed above belongs to the inner provider.

        A hand-maintained copy list is how ``max_tokens`` got lost -- and with it
        a thinking model's output budget, silently.  Delegating makes the next
        forgotten field a non-event instead of a bug: ``api_key``, ``base_url``
        and whatever else a caller inspects stay reachable through the cache.
        """
        return getattr(self.inner, item)

    def complete(self, messages: Sequence[Message], **kw: Any) -> LLMReply:
        key = prompt_fingerprint(self.name, self.model, messages) + "|%.2f|%d" % (
            float(kw.get("temperature", 0.2)),
            int(kw.get("max_tokens", 1200)),
        )
        if key in self._store:
            self.hits += 1
            reply = self._store[key]
            return LLMReply(**{**reply.__dict__, "cached": True})
        self.misses += 1
        reply = self.inner.complete(messages, **kw)
        if len(self._store) < self.max_entries:
            self._store[key] = reply
        return reply


class RecordingProvider(LLMProvider):
    """Replay: match on the last user message, fall back to a default reply."""

    def __init__(self, entries: Sequence[Mapping[str, Any]], *, name: str = "replay", default: str = "{}") -> None:
        self.name = name
        self.model = "replay"
        self.entries = [dict(e) for e in entries]
        self.default = default
        self.used: List[int] = []

    def complete(self, messages: Sequence[Message], **kw: Any) -> LLMReply:
        blob = "\n".join(m.content for m in messages)
        for idx, entry in enumerate(self.entries):
            needle = str(entry.get("match", ""))
            if needle and needle in blob:
                return LLMReply(text=str(entry.get("reply", "")), model=self.model, meta={"replay_index": idx})
        for idx, entry in enumerate(self.entries):
            if entry.get("match") in (None, "", "*") and idx not in self.used:
                self.used.append(idx)
                return LLMReply(text=str(entry.get("reply", "")), model=self.model, meta={"replay_index": idx})
        return LLMReply(text=self.default, model=self.model, meta={"replay_index": -1})


# --------------------------------------------------------------------------- #
# OpenAI-compatible HTTP provider (stdlib only)
# --------------------------------------------------------------------------- #


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 90.0,
        concurrency: int = 4,
        max_retries: int = 3,
        name: str = "openai-compat",
    ) -> None:
        self.name = name
        self.model = model
        # ``api_key=""`` means "this endpoint needs none" and must not fall back
        # to a hosted key from the environment: sending somebody's OpenAI key to
        # a local server is both wrong and a leak.  Only ``None`` means unset.
        self.api_key = api_key if api_key is not None else (os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = float(timeout)
        self.concurrency = int(concurrency)
        self.max_retries = int(max_retries)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMReply:
        if not self.api_key and self.require_key:
            raise LLMError("no API key: set OPENAI_API_KEY/LLM_API_KEY or pass --api-key")
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
        ).encode("utf-8")
        url = "%s/chat/completions" % self.base_url
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = "Bearer %s" % self.api_key
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                latency = round((time.time() - started) * 1000, 1)
                choice = (payload.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = payload.get("usage") or {}
                return LLMReply(
                    text=text,
                    model=payload.get("model", self.model),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=latency,
                    meta={"finish_reason": choice.get("finish_reason")},
                )
            except urllib.error.HTTPError as exc:  # 4xx: do not retry blindly
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:400]
                except Exception:  # pragma: no cover - diagnostics only
                    detail = ""
                last = LLMError("HTTP %s from %s: %s" % (exc.code, url, detail))
                if 400 <= int(exc.code) < 500 and int(exc.code) not in (408, 429):
                    break
            except Exception as exc:  # network/timeout: backoff and retry
                last = exc
            time.sleep(min(8.0, 0.8 * (2 ** attempt)))
        raise LLMError("provider %s failed after %d attempts: %s" % (self.name, self.max_retries, last))


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek's OpenAI-compatible endpoint.

    The default endpoint is applied when the caller passes nothing **or**
    ``None``.  A plain ``setdefault`` is not enough: ``make_provider`` forwards
    ``base_url=None`` for a channel that declares none, so the generic OpenAI
    fallback in the parent class would win -- posting a DeepSeek key to
    ``api.openai.com`` and getting a 401 back that says "Incorrect API key".
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, model: str = "deepseek-chat", **kw: Any) -> None:
        kw["base_url"] = (
            kw.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL") or self.DEFAULT_BASE_URL
        )
        kw.setdefault("name", "deepseek")
        super().__init__(model, **kw)


# --------------------------------------------------------------------------- #
# offline mock provider
# --------------------------------------------------------------------------- #

MARKER_RE = re.compile(
    r"#\s*(?P<kind>BUG|FIXME|HACK)\s*[:\-]?\s*(?P<name>[A-Za-z0-9_][A-Za-z0-9_./]*(?:-[A-Za-z0-9_./]+)*)"
    r"\s*(?:[:\-]{1,2}\s*(?P<note>.*))?",
    re.I,
)

#: The mock "engineer"'s repertoire: defect shape -> anchored ops.
#:
#: Kept deliberately dumb and explicit, because the mock exists to prove the
#: *plumbing*, not to reason.  Two properties make it useful on a repository it
#: has never seen:
#:
#: * entries carry only ``search``/``replace`` -- the file is taken from where
#:   the marker was actually reported, so the same recipe works in any layout;
#: * :func:`mock_playbook_for` matches a marker by its full name first and then
#:   by its last segment, so a defect named ``myapp/mean_zero`` still matches.
#:
#: A marker with no entry yields ``ops: []`` and a stated reason, and the loop
#: reports "could not express a fix" instead of fabricating an edit.
MOCK_PLAYBOOK: Dict[str, List[Dict[str, str]]] = {
    "text/title_case_off_by_one": [
        {
            "search": 'escaped = "".join(c if i > len(chars) - 2 else "\\\\" + c for i, c in enumerate(chars))',
            "replace": 'escaped = "".join(chars)',
            "rationale": "the escape loop corrupted the word; keep the characters as they are",
        },
        {
            "search": "out.append(escaped[:1].upper() + escaped[1:].lower())",
            "replace": 'out.append("-".join(part[:1].upper() + part[1:].lower() for part in escaped.split("-")))',
            "rationale": "capitalise every hyphen-separated part, as the docstring promises",
        },
    ],
    "stats/mean_zero": [
        {
            "search": "if not values:\n        return 0.0",
            "replace": 'if not values:\n        raise ValueError("mean() of empty sequence")',
            "rationale": "silent zero hides upstream data errors",
        }
    ],
    "mathx/clamp_inverted": [
        {
            "search": "    if value > low:\n        return low",
            "replace": "    if value < low:\n        return low",
            "rationale": "inverted comparison makes clamp() return the floor for every in-range value",
        }
    ],
}

#: Short aliases, so a repository that names its defects differently still gets
#: a fix when the *shape* of the defect is one the mock knows.
MOCK_PLAYBOOK_ALIASES: Dict[str, str] = {
    "mean_zero": "stats/mean_zero",
    "empty_mean": "stats/mean_zero",
    "clamp_inverted": "mathx/clamp_inverted",
    "title_case_off_by_one": "text/title_case_off_by_one",
}


def mock_playbook_for(marker: str) -> List[Dict[str, str]]:
    """Anchored ops for a marker, by full name, alias, or last path segment."""
    name = str(marker or "").strip()
    if not name:
        return []
    if name in MOCK_PLAYBOOK:
        return MOCK_PLAYBOOK[name]
    tail = name.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
    if tail in MOCK_PLAYBOOK:
        return MOCK_PLAYBOOK[tail]
    if tail in MOCK_PLAYBOOK_ALIASES:
        return MOCK_PLAYBOOK[MOCK_PLAYBOOK_ALIASES[tail]]
    return []


class MockProvider(LLMProvider):
    """Deterministic provider that drives the loop end to end without network."""

    name = "mock"
    model = "mock-grader"
    concurrency = 8

    def __init__(self, plan: Optional[Mapping[str, Any]] = None) -> None:
        self.plan = dict(plan or {})

    # -- helpers ----------------------------------------------------------- #
    def _code(self, messages: Sequence[Message]) -> Dict[str, str]:
        blob = "\n".join(m.content for m in messages)
        files: Dict[str, str] = {}
        pattern = re.compile(
            r"^===== FILE: (?P<path>.+?) =====(?: \[[^\]]*\])?\n(?P<body>.*?)(?=^===== FILE: |^===== MANIFEST|\Z)",
            re.S | re.M,
        )
        for match in pattern.finditer(blob):
            body = match.group("body")
            cut = body.find("\nReturn only the JSON object")
            if cut != -1:
                body = body[:cut]
            path = match.group("path").strip()
            if "[omitted" in body:
                continue
            files[path] = body + ("\n" if body and not body.endswith("\n") else "")
        return files

    def _role(self, messages: Sequence[Message]) -> str:
        blob = "\n".join(m.content for m in messages)
        found = re.findall(r"<agent_role>\s*([a-z0-9_\-]+)\s*</agent_role>", blob)
        return found[-1] if found else "unknown"

    def _json(self, payload: Any) -> LLMReply:
        return LLMReply(text=json.dumps(payload, ensure_ascii=False, indent=2), model=self.model, meta={"mock": True})

    # -- main -------------------------------------------------------------- #
    def complete(self, messages: Sequence[Message], *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMReply:
        role = self._role(messages)
        files = self._code(messages)
        markers: List[Dict[str, str]] = []
        for path, body in files.items():
            for line_no, line in enumerate(body.splitlines(), start=1):
                m = MARKER_RE.search(line)
                if m:
                    markers.append(
                        {
                            "name": m.group("name"),
                            "path": path,
                            "line": str(line_no),
                            "note": (m.group("note") or "").strip(),
                            "source": line.strip(),
                        }
                    )
        markers.sort(key=lambda m: (m["path"], int(m["line"])))

        if role == "test_sandbox":
            return self._json({"verdict": "approve", "observations": "sandbox executed", "confidence": 0.6})
        if role == "architecture_review":
            return self._json({"verdict": "approve", "confidence": 0.7, "blocking": [], "notes": "no structural objection raised by mock"})
        if role == "correctness_review":
            blob = "\n".join(m.content for m in messages)
            failing = '"failed": 0' in blob.replace(" ", "") or "pending" in blob.lower()
            verdict = "reject" if failing else "approve"
            return self._json(
                {
                    "verdict": verdict,
                    "confidence": 0.6,
                    "blocking": [{"issue_id": "*", "reason": "test evidence still reports failures"}] if failing else [],
                    "notes": "mock verdict follows the injected test evidence",
                }
            )
        if role == "risk_review":
            return self._json({"verdict": "approve", "blast_radius": "module", "risk": "low", "notes": "mock: blast radius bounded to the touched module"})
        if role == "edge_case_qa":
            return self._json({"verdict": "approve", "confidence": 0.6, "observations": "no new edge case found by mock", "blocking": []})
        if role == "refine_synthesize":
            issue_id = (re.findall(r"\"id\":\s*\"(ISS-\d+)\"", "\n".join(m.content for m in messages)) or ["ISS-0001"])[0]
            return self._json(
                {
                    "decision": "act",
                    "summary": "restore the documented contract for this defect",
                    "acceptance": ["the declared defect no longer reproduces", "existing test suite stays green"],
                    "falsification": "restore the defective line and the regression test must fail again",
                    "non_goals": ["no public API change"],
                    "blast_radius": "module",
                    "risk": "low",
                }
            )
        if role == "plan_split":
            paths = sorted({m["path"] for m in markers})
            return self._json(
                {
                    "steps": [
                        {"order": 1, "action": "apply the declared fix at the marked line", "files": paths or []},
                        {"order": 2, "action": "rerun the suite and confirm the marker is gone", "files": []},
                    ],
                    "estimate": "XS",
                    "rollback": "single hunk revert",
                }
            )
        if role == "test_design":
            return self._json(
                {
                    "tests": [
                        {
                            "kind": "regression",
                            "name": "test_marker_defect_%s" % (markers[0]["name"].replace("/", "_") if markers else "generic"),
                            "path": "tests/test_%s.py" % (markers[0]["name"].split("/")[-1] if markers else "regression"),
                            "given": "the marked input from the fixture",
                            "when": "the public function is called",
                            "then": "the documented result is returned",
                            "regression_of": markers[0]["name"] if markers else "",
                        }
                    ]
                }
            )
        if role in ("patch_generate", "patch_refactor"):
            if not markers:
                return self._json(
                    {
                        "ops": [],
                        "rationale": "mock provider found no '# BUG:' marker in the supplied files, so it refuses to invent a patch",
                        "risk": "unknown",
                        "expected_effect": "none",
                    }
                )
            marker = markers[0]
            ops = mock_playbook_for(marker["name"])
            if not ops:
                return self._json(
                    {
                        "ops": [],
                        "rationale": "the offline mock has no recipe for defect shape %r; "
                        "it will not invent an edit (use a real provider or extend "
                        "qedloop.llm.MOCK_PLAYBOOK)" % marker["name"],
                        "risk": "unknown",
                        "expected_effect": "none",
                        "bug_marker": marker["name"],
                    }
                )
            # The file comes from where the marker was reported, so the same
            # recipe works in any repository layout.
            ops = [{"path": marker["path"], **op} for op in ops]
            return self._json(
                {
                    "ops": ops,
                    "rationale": "mock playbook entry %r" % marker["name"],
                    "risk": "low",
                    "expected_effect": "the marked defect no longer reproduces",
                    "bug_marker": marker["name"],
                }
            )
        if role == "patch_reconcile":
            # The mock always picks the minimal candidate; the choice is stated
            # explicitly so a trace shows which candidate was taken.
            return self._json({"strategy": "minimal", "pick": "A", "reason": "mock prefers the minimal, verifiable patch"})
        if role in ("discover_archaeology", "discover_behavior", "discover_security"):
            return self._json({"issues": _mock_discovery(markers, "medium" if role == "discover_security" else "high")})
        return self._json({"note": "mock provider has no scripted answer for role %r" % role})


def _mock_discovery(markers: List[Dict[str, str]], severity: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for marker in markers[:3]:
        issues.append(
            {
                "title": "declared defect %s in %s" % (marker["name"], marker["path"]),
                "severity": severity,
                "confidence": 0.8,
                "files": [marker["path"]],
                "symbols": [],
                "evidence": "line %s: %s" % (marker["line"], marker["source"]),
                "why_it_matters": marker["note"] or "the marker asserts behavior the code does not implement",
                "proposed_direction": "fix the behavior at the marked line and lock it with a regression test",
                "bug_marker": marker["name"],
                "phase_hint": "patch",
            }
        )
    return issues


# --------------------------------------------------------------------------- #
# named channels
# --------------------------------------------------------------------------- #


@dataclass
class ProviderChannel:
    """A named model endpoint, declared once and selected by name.

    One repository usually talks to several models: a cheap one for discovery, a
    strong one for patching, a local one for offline work.  A channel records
    everything needed to reach one of them, so switching is a name rather than a
    wall of flags::

        providers:
          local:
            base_url: http://127.0.0.1:11434/v1
            model: qwen2.5-coder:14b
            api_key_env: ""            # an empty value means "no key needed"
          work:
            base_url: https://gateway.corp.example/v1
            model: gpt-4o-mini
            api_key_env: WORK_LLM_KEY
    """

    name: str
    kind: str = "openai-compat"
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None       # literal value; prefer api_key_env
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    cache: Optional[bool] = None
    max_tokens: Optional[int] = None
    note: str = ""
    #: May ``--provider auto`` choose this channel on its own?  A channel the
    #: user declared is fair game; a built-in preset for a local server that is
    #: probably not running is not -- auto-select them and every run on a fresh
    #: machine would hang on a dead port instead of falling back to the mock.
    auto_select: bool = True

    # -- key resolution ---------------------------------------------------- #
    def resolved_key(self, override: Optional[str] = None) -> Optional[str]:
        """CLI flag wins, then the channel's own value, then its env var."""
        if self.kind == "mock":
            # The mock makes no requests.  Borrowing whichever hosted key
            # happens to be in the environment would make every listing of
            # channels claim it has one.
            return None
        if override:
            return override
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None

    @property
    def requires_key(self) -> bool:
        """A local endpoint typically needs none; a hosted one always does.

        The mock is the one provider that can never need a key, so declaring
        ``api_key_env: ""`` on it must not make it look unconfigured.
        """
        return bool(self.api_key_env) and self.kind != "mock"

    def ready(self, override: Optional[str] = None) -> bool:
        return bool(self.resolved_key(override)) or not self.requires_key

    def to_dict(self, reveal_key: bool = False) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url or _default_base_url(self.kind),
            "model": self.model or _default_model(self.kind),
            "api_key_env": self.api_key_env or "",
            "key_present": bool(self.resolved_key()),
            "temperature": self.temperature,
            "timeout": self.timeout,
            "cache": self.cache,
            "max_tokens": self.max_tokens,
            "note": self.note,
        }
        if reveal_key:
            data["api_key"] = self.api_key
        return data


def _default_base_url(kind: str) -> str:
    if kind == "deepseek":
        return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )


def _default_model(kind: str) -> str:
    if kind == "deepseek":
        return "deepseek-chat"
    if kind == "mock":
        return "mock-grader"
    return os.environ.get("LLM_MODEL") or "gpt-4o-mini"


DEFAULT_CHANNELS: Dict[str, ProviderChannel] = {
    "openai": ProviderChannel(
        name="openai", kind="openai", model="gpt-4o-mini", api_key_env="OPENAI_API_KEY",
        note="any OpenAI-compatible endpoint; override base_url for a gateway",
    ),
    "deepseek": ProviderChannel(
        name="deepseek", kind="deepseek", model="deepseek-chat", api_key_env="DEEPSEEK_API_KEY",
        note="DeepSeek official API",
    ),
    "ollama": ProviderChannel(
        name="ollama", kind="openai-compat", base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:14b", api_key_env="", timeout=300.0, auto_select=False,
        note="local Ollama; no API key required",
    ),
    "lmstudio": ProviderChannel(
        name="lmstudio", kind="openai-compat", base_url="http://127.0.0.1:1234/v1",
        model="local-model", api_key_env="", timeout=300.0, auto_select=False,
        note="local LM Studio server; no API key required",
    ),
    "mock": ProviderChannel(
        name="mock", kind="mock", api_key_env="",
        note="deterministic offline provider: proves the plumbing, does not reason",
    ),
}

#: Built-in kind -> constructor.  ``openai-compat`` covers every vendor that
#: speaks ``POST {base_url}/chat/completions``.
PROVIDER_KINDS = ("openai", "openai-compat", "deepseek", "mock")


def channel_from_mapping(name: str, data: Mapping[str, Any]) -> ProviderChannel:
    """Normalise a ``providers:`` entry from a config file."""
    kind = str(data.get("kind") or data.get("type") or "openai-compat").strip().lower()
    if kind in ("openai_compat", "compat", "openai-compatible"):
        kind = "openai-compat"
    if kind not in PROVIDER_KINDS:
        raise LLMError(
            "channel %r has unknown kind %r (known: %s)" % (name, kind, ", ".join(PROVIDER_KINDS))
        )
    # A mock channel talks to nobody, so the generic OPENAI_API_KEY default
    # would only put a misleading "key from OPENAI_API_KEY" cell next to it in
    # `run.py channels` -- and report "yes" the moment that variable is set.
    default_env = "" if kind == "mock" else "OPENAI_API_KEY"
    api_key_env = data.get("api_key_env", data.get("key_env", default_env))
    if api_key_env is None:
        api_key_env = ""
    return ProviderChannel(
        name=name,
        kind=kind,
        base_url=(str(data["base_url"]).rstrip("/") if data.get("base_url") else None),
        model=(str(data["model"]) if data.get("model") else None),
        api_key=(str(data["api_key"]) if data.get("api_key") else None),
        api_key_env=str(api_key_env),
        temperature=(float(data["temperature"]) if data.get("temperature") is not None else None),
        timeout=(float(data["timeout"]) if data.get("timeout") is not None else None),
        cache=(bool(data["cache"]) if data.get("cache") is not None else None),
        max_tokens=(int(data["max_tokens"]) if data.get("max_tokens") is not None else None),
        note=str(data.get("note", "")),
    )


def normalize_channels(channels: Optional[Mapping[str, Any]]) -> Dict[str, ProviderChannel]:
    """Accept ``{name: ProviderChannel}`` or ``{name: mapping}`` from callers."""
    out: Dict[str, ProviderChannel] = {}
    for name, value in (channels or {}).items():
        if isinstance(value, ProviderChannel):
            out[str(name)] = value
        elif isinstance(value, Mapping):
            out[str(name)] = channel_from_mapping(str(name), value)
    return out


def known_channels(channels: Optional[Mapping[str, Any]] = None) -> Dict[str, ProviderChannel]:
    """Built-in channels plus the caller's own.

    A user-declared channel wins by name.  A built-in local-endpoint preset is
    also dropped when the caller points at the same ``base_url``, because showing
    both ``ollama`` and ``my-ollama`` for one server is noise, not choice.
    """
    custom = normalize_channels(channels)
    merged: Dict[str, ProviderChannel] = {}
    claimed = {channel.base_url for channel in custom.values() if channel.base_url}
    for name, channel in DEFAULT_CHANNELS.items():
        if channel.base_url and channel.base_url in claimed:
            continue
        merged[name] = channel
    merged.update(custom)
    return merged


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


def make_provider(
    spec: str = "auto",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    cache: bool = True,
    temperature: float = 0.2,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    channels: Optional[Mapping[str, Any]] = None,
) -> LLMProvider:
    """Resolve a provider name -- built-in or a configured channel -- to an instance.

    Precedence for each field: explicit argument > channel setting > environment
    > built-in default.  ``timeout=None`` means "not specified", not "90
    seconds": a channel that declares a 300 s timeout for a slow local model
    must get it.  ``auto`` uses the first *auto-selectable* ready channel (a
    channel is ready when its key is present, or when it declares that it needs
    none, as a local server does) and falls back to the mock, so
    ``python run.py run --target examples/buggy_service`` always works offline.
    """
    available = known_channels(channels)
    spec = (spec or "auto").strip()

    if spec.lower() == "auto":
        chosen = next(
            (
                channel
                for channel in available.values()
                if channel.kind != "mock" and channel.auto_select and channel.ready(api_key)
            ),
            available["mock"],
        )
    elif spec in available:
        chosen = available[spec]
    elif spec.lower() in available:
        chosen = available[spec.lower()]
    else:
        raise LLMError(
            "unknown provider %r (built-in: auto, %s; configured channels: %s)"
            % (spec, ", ".join(sorted(PROVIDER_KINDS)), ", ".join(sorted(available)) or "none")
        )

    resolved_model = model or chosen.model or _default_model(chosen.kind)
    resolved_base = base_url or chosen.base_url
    # An explicit argument wins; the channel's own setting is next; the shared
    # default is last.  Precedence by presence, not by value, or a channel that
    # asks for a 300 s local-model timeout would be silently given 90 s.
    resolved_timeout = timeout if timeout else (chosen.timeout or 90.0)
    resolved_temperature = chosen.temperature if chosen.temperature is not None else temperature
    # A channel that declares ``api_key_env: ""`` wants *no* credential, so it
    # gets the empty string rather than ``None``: None would let the parent
    # constructor reach for OPENAI_API_KEY/LLM_API_KEY in the environment and
    # send a hosted key to a local server.
    resolved_key = api_key or (chosen.resolved_key() if chosen.requires_key else "")

    if chosen.kind == "mock":
        provider: LLMProvider = MockProvider()
    elif chosen.kind == "deepseek":
        provider = DeepSeekProvider(
            resolved_model,
            api_key=resolved_key,
            base_url=resolved_base,
            timeout=resolved_timeout,
        )
    else:
        provider = OpenAICompatProvider(
            resolved_model,
            api_key=resolved_key,
            base_url=resolved_base,
            timeout=resolved_timeout,
            name=chosen.name,
        )
    provider.channel = chosen.name  # type: ignore[attr-defined]
    provider.require_key = chosen.requires_key  # type: ignore[attr-defined]
    provider.temperature = resolved_temperature  # type: ignore[attr-defined]
    provider.max_tokens = max_tokens or chosen.max_tokens  # type: ignore[attr-defined]
    provider.request_timeout = resolved_timeout  # type: ignore[attr-defined]

    use_cache = cache if chosen.cache is None else chosen.cache
    return CachingProvider(provider) if use_cache and chosen.kind != "mock" else provider
