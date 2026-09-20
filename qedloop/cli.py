# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Command line interface and user-facing formatting helpers.

Kept free of third-party dependencies so the framework is usable from a bare
interpreter:

    python run.py run --target examples/buggy_service --provider mock
    python run.py apply --run runs/<run_id> --dry-run
    python run.py agents
    python run.py check --target examples/buggy_service
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import ConfigError, deep_merge, load_config
from .core import detect_markers, render_markers
from .crew import agent_catalog
from .llm import PROVIDER_KINDS, LLMError, Message, known_channels, make_provider
from .orchestrator import RunConfig, apply_run, load_codebase, run_loop
from .policy import Policy
from .sandbox import pytest_available, run_pytest

PROVIDERS = ("auto", "openai", "openai-compat", "deepseek", "mock")


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, max_width: int = 72) -> str:
    columns = len(headers)
    cells = [[_clip(str(cell), max_width) for cell in row] + [""] * (columns - len(row)) for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for index in range(columns):
            widths[index] = max(widths[index], len(row[index]))
    line = "  ".join("-" * width for width in widths)
    out = ["  ".join(headers[i].ljust(widths[i]) for i in range(columns)), line]
    for row in cells:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(columns)).rstrip())
    return "\n".join(out)


def _clip(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def phase_banner(name: str, agent_count: int) -> str:
    return "== %s (%d agents) ==" % (name, agent_count)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="AI-driven, self-iterating software development loop (5 phases x 3 agents).",
    )
    parser.add_argument("--version", action="store_true", help="print the framework version and exit")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the loop against a repository")
    run.add_argument("--target", default=None, help="repository to iterate over (default: examples/buggy_service)")
    run.add_argument("--config", default=None, help="YAML/JSON config file (see examples/loop.buggy_service.yml)")
    run.add_argument("--out", default=None, help="directory for run artifacts (default: runs)")
    run.add_argument("--max-iterations", type=int, default=None, help="phase-5 loop budget (default: 3)")
    run.add_argument("--max-steps", type=int, default=None, help="hard node-execution budget (default: 64)")
    run.add_argument(
        "--provider", default=None,
        help="model channel: auto | a built-in kind (%s) | a name from the config's providers: section" % ", ".join(PROVIDER_KINDS),
    )
    run.add_argument("--model", default=None, help="model id for the chosen channel")
    run.add_argument("--api-key", default=None, help="API key (overrides the channel's api_key_env)")
    run.add_argument("--base-url", default=None, help="custom API base URL, e.g. https://gateway.corp/v1")
    run.add_argument("--llm-timeout", type=float, default=None, help="per-request timeout in seconds (default: 90)")
    run.add_argument(
        "--brief",
        default=None,
        help="file of project constraints sent to every agent "
             "(relative to the config file, or to the working directory without --config)",
    )
    run.add_argument("--include", default=None, help="comma-separated globs, e.g. '*.py,*.md'")
    run.add_argument("--exclude", default=None, help="comma-separated globs to skip")
    run.add_argument("--quality-target", type=float, default=None, help="phase-5 quality threshold (default: 0.80)")
    run.add_argument("--min-approvals", type=int, default=None, help="review approvals needed (default: 1)")
    run.add_argument("--max-self-loops", type=int, default=None, help="review->refine budget (default: 3)")
    run.add_argument("--token-budget", type=int, default=None, help="stop calling agents past this many tokens")
    run.add_argument("--no-tests", action="store_true", help="do not execute the target test suite")
    run.add_argument("--test-timeout", type=float, default=None, help="seconds per pytest invocation (default: 120)")
    run.add_argument("--single-cycle", action="store_true", help="never loop; one pass then report")
    run.add_argument("--dry-run", action="store_true", help="do not touch the target repository")
    run.add_argument("--quiet", action="store_true", help="write artifacts but suppress live output")
    run.add_argument("--run-id", default=None, help="explicit run id (default: timestamp)")

    apply_cmd = sub.add_parser("apply", help="write a run's verified candidate tree onto the target")
    apply_cmd.add_argument("--run", required=True, help="run directory, e.g. runs/20250101-120000-0001")
    apply_cmd.add_argument("--target", default=None, help="override the target repository")
    apply_cmd.add_argument("--dry-run", action="store_true", help="report what would change")
    apply_cmd.add_argument("--no-backup", action="store_true", help="skip .qedloop.bak files")
    apply_cmd.add_argument("--allow-unverified", action="store_true", help="apply even if the run did not converge")

    sub.add_parser("agents", help="list every agent role in the crew")

    channels = sub.add_parser("channels", help="list model channels (built-in and configured)")
    channels.add_argument("--config", default=None, help="config file whose providers: section to include")
    channels.add_argument("--probe", action="store_true", help="send one round-trip per ready channel to verify it")
    channels.add_argument("--probe-timeout", type=float, default=8.0, help="probe timeout in seconds (default: 8)")

    check = sub.add_parser("check", help="baseline diagnostics for a target repository")
    check.add_argument("--target", default="examples/buggy_service")
    check.add_argument("--include", default="*.py")
    check.add_argument("--no-tests", action="store_true")
    return parser


# --------------------------------------------------------------------------- #
# config resolution
# --------------------------------------------------------------------------- #


def resolve_config(args: argparse.Namespace) -> RunConfig:
    file_config: Dict[str, Any] = {}
    if args.config:
        file_config = load_config(args.config)
    merged = deep_merge(file_config, {})
    run_section = dict(merged.get("run") or {})
    flat = {**run_section, **{k: v for k, v in merged.items() if k not in ("run", "gate", "policy")}}
    flat["gate"] = merged.get("gate") or merged.get("policy") or flat.get("gate") or {}
    if args.brief:
        # Merged into the mapping rather than passed as an override: the brief
        # key means "path", and RunConfig.from_mapping is the single place that
        # turns it into text.
        flat["brief"] = args.brief

    target = args.target or flat.get("target") or "examples/buggy_service"
    if not args.config and not args.target and not Path(target).is_dir():
        target = "."

    overrides: Dict[str, Any] = {
        "target": target,
        "out_dir": args.out,
        "max_iterations": args.max_iterations,
        "max_steps": args.max_steps,
        "provider": args.provider,
        "model": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "llm_timeout": args.llm_timeout,
        "token_budget": args.token_budget,
        "test_timeout": args.test_timeout,
        "run_tests": False if args.no_tests else bool(flat.get("run_tests", True)),
        "allow_multi_cycle": False if args.single_cycle else flat.get("allow_multi_cycle", True),
        "dry_run": True if args.dry_run else bool(flat.get("dry_run", False)),
        "quiet": True if args.quiet else bool(flat.get("quiet", False)),
        "run_id": args.run_id,
        "cache": flat.get("cache", True),
    }
    if args.include:
        overrides["include"] = tuple(part.strip() for part in args.include.split(",") if part.strip())
    if args.exclude:
        overrides["exclude"] = tuple(part.strip() for part in args.exclude.split(",") if part.strip())

    policy_data = dict(flat.get("gate") or {})
    if args.quality_target is not None:
        policy_data["quality_target"] = args.quality_target
    if args.min_approvals is not None:
        policy_data["min_approvals"] = args.min_approvals
    if args.max_self_loops is not None:
        policy_data["max_self_loops"] = args.max_self_loops

    config = RunConfig.from_mapping(
        {**flat, "gate": policy_data},
        brief_base=str(Path(args.config).resolve().parent) if args.config else None,
        **overrides,
    )
    if args.no_tests:
        config.run_tests = False
    for env_key, attr in (("LLM_PROVIDER", "provider"), ("LLM_MODEL", "model"), ("LLM_BASE_URL", "base_url")):
        if os.environ.get(env_key) and not getattr(config, attr):
            setattr(config, attr, os.environ[env_key])
    return config


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    try:
        config = resolve_config(args)
    except ConfigError as exc:
        print("config error: %s" % exc, file=sys.stderr)
        return 2
    if config.quiet:
        # `quiet` suppresses the live trace, not the fact that a run happened.
        # One cycle against a real model takes minutes, and printing nothing for
        # that whole time is indistinguishable from a hang -- so the run says
        # where it is and where to watch, and reports its result at the end.
        if not config.run_id:
            config.run_id = config.resolved_run_id()
        print("running : %s" % config.target)
        print("model   : %s / %s" % (config.provider, config.model or "channel default"))
        print("progress: Get-Content %s -Tail 20" % (Path(config.out_dir) / config.run_id / "trace.jsonl"))
    try:
        result = run_loop(config)
    except LLMError as exc:
        print("model channel error: %s" % exc, file=sys.stderr)
        print("", file=sys.stderr)
        print("run 'python run.py channels --config <file>' to see what is available", file=sys.stderr)
        return 2
    if config.quiet:
        print("")
        for line in (result.report.summary_lines() if result.report else result.summary_lines()):
            print(line)
    if result.status != "converged" and not config.quiet:
        print("")
        print("next: inspect %s, then 'python run.py apply --run %s --dry-run'" % (result.run_dir / "REPORT.md", result.run_dir))
    return result.exit_code


def cmd_apply(args: argparse.Namespace) -> int:
    outcome = apply_run(
        args.run,
        target=args.target,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        allow_unverified=args.allow_unverified,
    )
    if outcome.get("reason"):
        print(outcome["reason"])
        return 1 if not outcome.get("ok") else 0
    for path in outcome.get("applied") or []:
        print("%s %s" % ("would write" if args.dry_run else "wrote     ", path))
    for item in outcome.get("skipped") or []:
        print("skipped   %s (%s)" % (item.get("path"), item.get("reason")))
    print("")
    print("%d file(s) %s in %s" % (len(outcome.get("applied") or []), "to change" if args.dry_run else "changed", outcome.get("target")))
    return 0 if outcome.get("ok") or args.dry_run else 1


def cmd_channels(args: argparse.Namespace) -> int:
    """List every channel that ``--provider`` would accept, and where it points.

    Keys are never printed: only whether one was found and where it came from.
    """
    configured: Dict[str, Any] = {}
    run_section: Dict[str, Any] = {}
    if args.config:
        try:
            file_config = load_config(args.config)
        except ConfigError as exc:
            print("config error: %s" % exc, file=sys.stderr)
            return 2
        configured = dict(file_config.get("providers") or file_config.get("channels") or {})
        run_section = dict(file_config.get("run") or {})

    channels = known_channels(configured)
    rows = []
    for name in sorted(channels):
        channel = channels[name]
        key = channel.resolved_key()
        rows.append([
            name,
            channel.kind,
            "yes" if key else ("not needed" if not channel.requires_key else "-- missing"),
            channel.api_key_env or "-",
            channel.model or _default_model_label(channel.kind),
            str(channel.max_tokens) if channel.max_tokens else "-",
            channel.base_url or _default_base_url_label(channel.kind),
        ])
    print("model channels ('--provider <name>')")
    print("")
    print(render_table(
        ["name", "kind", "key", "key from", "model", "max_tokens", "base_url"], rows, max_width=48,
    ))
    print("")
    print("sources: built-in channels, overridden by your config's providers: section")
    if configured:
        print("config : %s (%d channel(s) declared)" % (args.config, len(configured)))
    else:
        print("config : none given (--config <file> to add your own)")
    effective = _effective_endpoint(channels, run_section)
    if effective:
        print("run    : --provider %s -> model=%s base_url=%s (the run: section wins at call time)"
              % effective)
    print("environ: OPENAI_API_KEY / LLM_API_KEY / DEEPSEEK_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_PROVIDER")

    if args.probe:
        print("")
        print("probing ready channels (one round-trip each, timeout %.0fs) ..." % args.probe_timeout)
        selected = str(run_section.get("provider") or "").strip()
        for name in sorted(channels):
            channel = channels[name]
            if not channel.ready():
                print("  %-14s skipped (no key)" % name)
                continue
            provider = None
            try:
                overrides: Dict[str, Any] = {}
                if name == selected:
                    # Probe what the run will actually call.  The `run:` section
                    # wins over the channel at call time, so probing the bare
                    # declaration can test an endpoint the run never uses --
                    # which is how a 401 from the wrong host looks like a bad key.
                    overrides = {
                        "model": run_section.get("model") or None,
                        "base_url": run_section.get("base_url") or None,
                    }
                provider = make_provider(
                    name, channels=channels, cache=False, timeout=args.probe_timeout, **overrides
                )
                reply = provider.complete(
                    [Message("user", 'Reply with exactly {"ok": true} and nothing else.')],
                    temperature=0.0,
                    max_tokens=32,
                )
                head = (reply.text or "").strip().replace("\n", " ")[:60]
                print("  %-14s ok   model=%s  at %s  reply=%s" % (
                    name, reply.model, getattr(provider, "base_url", "") or "-", head or "(empty)",
                ))
            except Exception as exc:  # probe failures are the point of the command
                endpoint = getattr(provider, "base_url", "") or ""
                print("  %-14s FAIL %s%s" % (
                    name, str(exc)[:130], ("  [%s]" % endpoint) if endpoint else "",
                ))
    return 0


def _default_model_label(kind: str) -> str:
    return "mock-grader" if kind == "mock" else ("deepseek-chat" if kind == "deepseek" else "gpt-4o-mini")


def _effective_endpoint(
    channels: Mapping[str, Any],
    run_section: Mapping[str, Any],
) -> Optional[Tuple[str, str, str]]:
    """What the run will actually call, when ``run:`` overrides a channel.

    The table above lists channel *declarations*; ``run.model`` and
    ``run.base_url`` win at call time.  Showing only the declarations is how a
    config looks right while sending requests to a different place -- the same
    precedence trap ``make_provider`` documents, moved to where it is visible.
    """
    name = str(run_section.get("provider") or "").strip()
    model = str(run_section.get("model") or "").strip()
    base_url = str(run_section.get("base_url") or "").strip()
    if not (name and (model or base_url)):
        return None
    channel = channels.get(name) or channels.get(name.lower())
    if channel is None:
        return None
    return (
        name,
        model or channel.model or _default_model_label(channel.kind),
        base_url or channel.base_url or _default_base_url_label(channel.kind),
    )


def _default_base_url_label(kind: str) -> str:
    if kind == "mock":
        return "-"
    if kind == "deepseek":
        return "https://api.deepseek.com/v1"
    return os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"


def cmd_agents(_: argparse.Namespace) -> int:
    catalog = agent_catalog()
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for row in catalog:
        by_phase.setdefault(str(row["phase"]), []).append(row)
    order = ["discover", "refine", "review", "patch", "qa"]
    for index, phase in enumerate(order, start=1):
        rows = by_phase.get(phase, [])
        print("")
        print(phase_banner("Phase %d - %s" % (index, phase), len(rows)))
        print(render_table(
            ["agent", "lens", "mode", "temp", "purpose"],
            [[r["role"], r["lens"], r["mode"], r["temperature"], r["description"]] for r in rows],
        ))
    print("")
    print("total: %d agents" % len(catalog))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    patterns = tuple(part.strip() for part in (args.include or "*.py").split(",") if part.strip())
    codebase = load_codebase(args.target, patterns)
    print("target        : %s" % codebase.root)
    print("revision      : %s" % codebase.revision)
    print("files         : %d (code %d, tests %d)" % (len(codebase.files), len(codebase.code_files), len(codebase.test_files)))
    print("")
    print(codebase.summary())
    markers = detect_markers(codebase.as_map())
    print("")
    print("declared defect markers: %d (actionable %d)" % (len(markers), len([m for m in markers if m.actionable])))
    print(render_markers(markers))
    print("")
    print("pytest importable: %s" % ("yes" if pytest_available() else "no"))
    if not args.no_tests and codebase.test_files:
        # Measured in a copy of the repository, like every other measurement in
        # this framework: materialising only the matched ``.py`` files drops the
        # images, fixtures and bundles the suite reads, and reports failures the
        # real repository does not have.
        report = run_pytest(codebase.as_map(), target_root=codebase.root, timeout=120.0)
        print("baseline suite: %s" % ("green" if report.green else "RED"))
        print("  %d passed, %d failed, %d errors (of the target's own tests)" % (report.passed, report.failed, report.errors))
        if report.error:
            print("  note: %s" % report.error)
        elif report.stdout:
            print("  " + _clip(report.stdout.strip().splitlines()[-1] if report.stdout.strip() else "", 120))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print("qedloop %s" % __version__)
        return 0
    if not args.command:
        parser.print_help()
        return 0
    handlers = {"run": cmd_run, "apply": cmd_apply, "agents": cmd_agents, "check": cmd_check, "channels": cmd_channels}
    return handlers[args.command](args)
