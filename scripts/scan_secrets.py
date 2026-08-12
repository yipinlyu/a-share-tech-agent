#!/usr/bin/env python3
"""Fail safely when repository text contains likely live credentials."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        "cache",
        "artifacts",
        "runtime",
        "htmlcov",
        "node_modules",
    }
)
PLACEHOLDER_MARKERS = (
    "replace-with",
    "placeholder",
    "example",
    "test-only",
    "test_",
    "test-",
    "fake",
    "dummy",
    "mock",
    "sample",
    "changeme",
    "your-",
    "do-not-leak",
    "not-a-secret",
    "never-trace",
    "offline-fixture",
    "environment-",
    "streamlit-",
)

_DEEPSEEK_KEY = re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{20,}")
_TUSHARE_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])")
_BEARER_TOKEN = re.compile(
    r"(?i)\bauthorization\s*[:=]\s*[\"']?bearer\s+([A-Za-z0-9._~+/=-]{16,})"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?ix)\b[A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|ACCESS[_-]?KEY)"
    r"[A-Z0-9_]*\s*=\s*[\"']?([^\s\"';,#}{]{16,})"
)


@dataclass(frozen=True, order=True)
class Finding:
    """A redacted credential finding safe to print in logs."""

    path: str
    line_number: int
    kind: str


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _classify_line(line: str) -> set[str]:
    findings: set[str] = set()
    for kind, pattern in (
        ("DEEPSEEK_KEY", _DEEPSEEK_KEY),
        ("TUSHARE_TOKEN", _TUSHARE_TOKEN),
        ("BEARER_TOKEN", _BEARER_TOKEN),
        ("SECRET_ASSIGNMENT", _SECRET_ASSIGNMENT),
    ):
        for match in pattern.finditer(line):
            candidate = match.group(1) if match.lastindex else match.group(0)
            if not _is_placeholder(candidate):
                findings.add(kind)
    return findings


def _tracked_files(root: Path) -> list[Path] | None:
    """Return Git-tracked files, or None when root is not a repository."""

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return [root / Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def _candidate_files(root: Path) -> Iterable[Path]:
    tracked = _tracked_files(root)
    paths = tracked if tracked is not None else root.rglob("*")
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in SKIPPED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file():
            yield path


def scan_path(root: Path | str) -> list[Finding]:
    """Scan tracked text files below root without retaining credential values."""

    root = Path(root).resolve()
    findings: list[Finding] = []
    for path in _candidate_files(root):
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if b"\0" in payload:
            continue
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(payload.decode("utf-8", errors="replace").splitlines(), 1):
            for kind in _classify_line(line):
                findings.append(Finding(relative, line_number, kind))
    return sorted(set(findings))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan repository text for likely live secrets.")
    parser.add_argument("path", nargs="?", default=".", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    findings = scan_path(args.path)
    for finding in findings:
        print(f"{finding.path}:{finding.line_number}: {finding.kind}")
    if findings:
        print(f"Secret scan failed: {len(findings)} redacted finding(s).")
        return 1
    print("Secret scan passed: no likely live credentials found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
