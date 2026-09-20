# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Model-channel tests: custom base URLs, named channels, key handling.

The framework must talk to any OpenAI-compatible endpoint -- a corporate
gateway, a local server, a third-party aggregator -- without a code change, and
it must never print a key it was given.
"""

from __future__ import annotations

import pytest

from qedloop.llm import (
    DEFAULT_CHANNELS,
    LLMError,
    Message,
    OpenAICompatProvider,
    ProviderChannel,
    channel_from_mapping,
    known_channels,
    make_provider,
    normalize_channels,
)


# --------------------------------------------------------------------------- #
# declaring a channel
# --------------------------------------------------------------------------- #


def test_channel_from_mapping_reads_every_field():
    channel = channel_from_mapping(
        "work",
        {
            "kind": "openai-compat",
            "base_url": "https://gateway.corp.example/v1/",
            "model": "gpt-4o-mini",
            "api_key_env": "CORP_KEY",
            "timeout": 300,
            "temperature": 0.1,
            "cache": False,
            "max_tokens": 2048,
            "note": "ask platform team",
        },
    )
    assert channel.name == "work"
    assert channel.base_url == "https://gateway.corp.example/v1", "trailing slash is normalised away"
    assert channel.model == "gpt-4o-mini"
    assert channel.api_key_env == "CORP_KEY"
    assert channel.timeout == 300.0
    assert channel.temperature == 0.1
    assert channel.cache is False
    assert channel.max_tokens == 2048


def test_channel_kind_aliases_are_accepted():
    for alias in ("compat", "openai-compatible", "openai_compat", "openai-compat"):
        assert channel_from_mapping("x", {"kind": alias}).kind == "openai-compat"


def test_unknown_channel_kind_is_rejected_with_the_known_list():
    with pytest.raises(LLMError) as excinfo:
        channel_from_mapping("bad", {"kind": "grpc"})
    assert "unknown kind" in str(excinfo.value) and "openai-compat" in str(excinfo.value)


def test_local_channel_needs_no_key_and_hosted_channel_does():
    local = channel_from_mapping("local", {"base_url": "http://127.0.0.1:11434/v1", "api_key_env": ""})
    hosted = channel_from_mapping("hosted", {"base_url": "https://api.example.com/v1", "api_key_env": "MY_KEY"})
    assert local.requires_key is False and local.ready() is True
    assert hosted.requires_key is True and hosted.ready() is False


def test_mock_never_looks_like_it_is_missing_a_key():
    """The built-in mock declares an env var for the hosted case; it must still
    report as ready, or `channels` would show a false alarm."""
    assert DEFAULT_CHANNELS["mock"].ready() is True
    assert normalize_channels({"m": {"kind": "mock"}})["m"].ready() is True


def test_mock_never_borrows_a_hosted_key(monkeypatch):
    """A mock makes no requests, so it must never claim to have a key.

    The generic ``OPENAI_API_KEY`` default for an undeclared ``api_key_env``
    used to make every ``run.py channels`` listing show the mock next to
    whichever hosted key happened to be exported.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-mine")
    mock = channel_from_mapping("m", {"kind": "mock"})
    assert mock.api_key_env == "", "a mock channel has no key source to declare"
    assert mock.resolved_key() is None, "and it must not borrow the environment's"
    assert mock.ready() is True


def test_a_channel_output_budget_survives_the_cache_wrapper():
    """``max_tokens`` is the model's output ceiling, not a prompt-cache detail.

    A thinking model spends part of that ceiling on reasoning tokens, so losing
    it in the wrapper silently cuts every reply off at the framework's small
    JSON-answer budget.
    """
    provider = make_provider(
        "local",
        channels={"local": {"base_url": "http://127.0.0.1:1234/v1", "model": "m",
                            "api_key_env": "", "max_tokens": 16000, "cache": True}},
    )
    assert provider.max_tokens == 16000
    assert provider.name.endswith("+cache"), "precondition: the cache wrapper is in play"


# --------------------------------------------------------------------------- #
# "not set" versus "explicitly empty"
# --------------------------------------------------------------------------- #


def test_deepseek_without_a_declared_base_url_does_not_go_to_openai(monkeypatch):
    """A hostless DeepSeek channel must reach DeepSeek, not api.openai.com.

    ``make_provider`` forwards ``base_url=None`` for a channel that declares
    none; a ``setdefault`` in the constructor therefore does nothing and the
    generic OpenAI fallback wins.  The request then carries a DeepSeek key to
    OpenAI and comes back ``401 Incorrect API key``, which looks like a bad key
    rather than the wrong host.
    """
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    provider = make_provider("deepseek", api_key="sk-test")
    assert provider.base_url == "https://api.deepseek.com/v1"

    declared = make_provider("mine", channels={"mine": {"kind": "deepseek", "api_key_env": "K"}}, api_key="sk-test")
    assert declared.base_url == "https://api.deepseek.com/v1"


def test_a_channel_declaring_no_key_never_borrows_a_hosted_one(monkeypatch):
    """``api_key_env: ""`` means "no credential", not "look one up".

    Local servers were unusable before this: the empty declaration fell back to
    the ambient OPENAI_API_KEY and the request failed with "no API key" -- and
    if that variable *was* set, the local server would have received it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-somebody-elses")
    provider = make_provider(
        "local",
        channels={"local": {"base_url": "http://127.0.0.1:11434/v1", "model": "m", "api_key_env": ""}},
    )
    assert provider.api_key == "", "a hosted key must not leak to a local endpoint"
    assert provider.require_key is False


def test_a_local_request_sends_no_authorization_header(monkeypatch):
    """The header is the part that actually leaks; assert on the wire."""
    import urllib.request

    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "{}"}}], "usage": {}}'

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        return _Response()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-somebody-elses")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    provider = make_provider(
        "local",
        channels={"local": {"base_url": "http://127.0.0.1:11434/v1", "model": "m", "api_key_env": ""}},
    )
    provider.complete([Message("user", "hi")])

    assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert "authorization" not in captured["headers"], captured["headers"]


def test_probe_uses_the_endpoint_the_run_will_actually_call(tmp_path, monkeypatch):
    """``--probe`` is a preflight for the run, so the run: section must apply.

    Probing the bare channel declaration tests an endpoint the run never uses --
    the 401 then comes from a host the user never configured.
    """
    from argparse import Namespace

    from qedloop.cli import cmd_channels
    from qedloop.llm import LLMReply

    config = tmp_path / "loop.yml"
    config.write_text(
        "run:\n"
        "  provider: mine\n"
        "  model: run-model\n"
        "  base_url: https://run.example/v1\n"
        "providers:\n"
        "  mine:\n"
        "    kind: openai-compat\n"
        "    model: channel-model\n"
        "    base_url: https://channel.example/v1\n"
        '    api_key_env: ""\n',
        encoding="utf-8",
    )

    calls = []

    class _Stub:
        base_url = "https://run.example/v1"

        def complete(self, messages, **kw):
            return LLMReply(text='{"ok": true}', model="stub")

    def _fake_make_provider(name, **kw):
        calls.append((name, kw))
        return _Stub()

    monkeypatch.setattr("qedloop.cli.make_provider", _fake_make_provider)
    assert cmd_channels(Namespace(config=str(config), probe=True, probe_timeout=1.0)) == 0

    probed = {name: kw for name, kw in calls}
    assert probed["mine"]["model"] == "run-model"
    assert probed["mine"]["base_url"] == "https://run.example/v1"


def test_key_resolution_precedence(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    channel = ProviderChannel(name="c", api_key_env="MY_KEY")
    assert channel.resolved_key() == "from-env"
    assert channel.resolved_key("from-flag") == "from-flag", "an explicit key wins over the environment"
    explicit = ProviderChannel(name="c", api_key="literal", api_key_env="MY_KEY")
    assert explicit.resolved_key() == "literal", "the channel's own value wins over the environment"


def test_channel_dict_never_reveals_the_key_by_default(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "sk-do-not-print-me")
    channel = ProviderChannel(name="c", api_key_env="SECRET_KEY")
    payload = channel.to_dict()
    assert payload["key_present"] is True
    assert "sk-do-not-print-me" not in repr(payload)
    assert channel.to_dict(reveal_key=True)["api_key"] is None  # only a literal api_key is ever echoed


# --------------------------------------------------------------------------- #
# resolving a provider from channels
# --------------------------------------------------------------------------- #


def test_make_provider_uses_the_channels_base_url_and_model(monkeypatch):
    monkeypatch.setenv("CORP_KEY", "secret-value")
    provider = make_provider(
        "work",
        channels={"work": {"base_url": "https://gateway.corp.example/v1", "model": "corp-model", "api_key_env": "CORP_KEY"}},
        cache=False,
    )
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.base_url == "https://gateway.corp.example/v1"
    assert provider.model == "corp-model"
    assert provider.api_key == "secret-value"
    assert provider.channel == "work"


def test_explicit_arguments_override_the_channel(monkeypatch):
    monkeypatch.setenv("CORP_KEY", "secret-value")
    provider = make_provider(
        "work",
        model="other-model",
        base_url="https://other.example/v1",
        api_key="flag-key",
        channels={"work": {"base_url": "https://gateway.corp.example/v1", "model": "corp-model", "api_key_env": "CORP_KEY"}},
        cache=False,
    )
    assert provider.base_url == "https://other.example/v1"
    assert provider.model == "other-model"
    assert provider.api_key == "flag-key"


def test_channel_timeout_is_used_for_local_models():
    provider = make_provider(
        "slow-local",
        channels={"slow-local": {"base_url": "http://127.0.0.1:11434/v1", "model": "m", "api_key_env": "", "timeout": 300}},
        cache=False,
    )
    assert provider.request_timeout == 300.0


def test_auto_skips_channels_that_are_not_ready(monkeypatch):
    """`auto` must not pick a hosted channel whose key is absent; falling back to
    the mock keeps `python run.py run` working on a fresh machine."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider = make_provider("auto", channels={"hosted": {"base_url": "https://x/v1", "api_key_env": "NOT_SET_ANYWHERE"}})
    assert provider.channel == "mock"


def test_auto_prefers_a_ready_configured_channel(monkeypatch):
    monkeypatch.setenv("CORP_KEY", "k")
    provider = make_provider(
        "auto",
        channels={"corp": {"base_url": "https://gateway.corp.example/v1", "model": "m", "api_key_env": "CORP_KEY"}},
        cache=False,
    )
    assert provider.channel == "corp" and provider.base_url == "https://gateway.corp.example/v1"


def test_auto_accepts_a_local_channel_without_a_key():
    provider = make_provider(
        "auto",
        channels={"local": {"base_url": "http://127.0.0.1:11434/v1", "model": "m", "api_key_env": ""}},
        cache=False,
    )
    assert provider.channel == "local"


def test_unknown_provider_lists_what_is_available():
    with pytest.raises(LLMError) as excinfo:
        make_provider("nope", channels={"work": {"base_url": "https://x/v1"}})
    message = str(excinfo.value)
    assert "unknown provider" in message and "work" in message and "ollama" in message


def test_builtin_local_preset_is_hidden_when_the_config_points_at_the_same_url():
    channels = known_channels({"mine": {"base_url": "http://127.0.0.1:11434/v1", "api_key_env": ""}})
    assert "mine" in channels
    assert "ollama" not in channels, "one server should appear once"
    assert "lmstudio" in channels, "an unrelated preset is still offered"


def test_configured_channel_overrides_a_builtin_of_the_same_name(monkeypatch):
    monkeypatch.setenv("CORP_KEY", "k")
    provider = make_provider(
        "deepseek",
        channels={"deepseek": {"kind": "openai-compat", "base_url": "https://mirror.example/v1", "model": "mirror", "api_key_env": "CORP_KEY"}},
        cache=False,
    )
    assert provider.base_url == "https://mirror.example/v1" and provider.model == "mirror"


def test_caching_wrapper_preserves_the_channel_identity():
    provider = make_provider(
        "work",
        channels={"work": {"base_url": "https://x/v1", "model": "m", "api_key_env": ""}},
        cache=True,
    )
    assert provider.channel == "work"
    assert provider.name.endswith("+cache")
    assert provider.request_timeout == provider.inner.request_timeout


# --------------------------------------------------------------------------- #
# HTTP request shape
# --------------------------------------------------------------------------- #


def test_openai_compat_provider_posts_to_the_configured_base_url(monkeypatch):
    """The whole point of a custom base URL: the request must go there."""
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            import json

            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        import json

        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"model": "m", "choices": [{"message": {"content": '{"ok": true}'}}], "usage": {}})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatProvider("m", api_key="k", base_url="https://gateway.corp.example/v1", timeout=42.0)
    reply = provider.complete([Message("user", "hi")], temperature=0.0, max_tokens=16)

    assert captured["url"] == "https://gateway.corp.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m" and captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["timeout"] == 42.0
    assert reply.text == '{"ok": true}'


def test_missing_key_fails_loudly_with_an_actionable_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider = OpenAICompatProvider("m", api_key=None, base_url="https://x/v1")
    with pytest.raises(LLMError) as excinfo:
        provider.complete([Message("user", "hi")])
    assert "API key" in str(excinfo.value)
