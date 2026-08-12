from __future__ import annotations

from pathlib import Path

import pytest

from scripts.scan_secrets import main, scan_path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_clean_tree_allows_documented_placeholders_and_skips_runtime_directories(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "config.example.toml",
        'TUSHARE_TOKEN = "replace-with-your-tushare-token"\n'
        'DEEPSEEK_API_KEY = "replace-with-your-deepseek-key"\n',
    )
    _write(tmp_path / "README.md", "Authorization: Bearer example-placeholder\n")

    live_key = "sk-" + "A" * 32
    for directory in (
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        "cache",
        "artifacts",
        "runtime",
    ):
        _write(tmp_path / directory / "ignored.txt", live_key)

    assert scan_path(tmp_path) == []


@pytest.mark.parametrize(
    ("filename", "content", "expected_kind"),
    [
        ("deepseek.env", 'DEEPSEEK_API_KEY="' + "sk-" + "A" * 32 + '"\n', "DEEPSEEK_KEY"),
        ("tushare.txt", "B" * 64 + "\n", "TUSHARE_TOKEN"),
        ("headers.txt", "Authorization: Bearer " + "C" * 32 + "\n", "BEARER_TOKEN"),
        ("settings.toml", 'client_secret = "' + "D" * 24 + '"\n', "SECRET_ASSIGNMENT"),
    ],
)
def test_scan_path_detects_supported_secret_shapes(
    tmp_path: Path,
    filename: str,
    content: str,
    expected_kind: str,
) -> None:
    _write(tmp_path / filename, content)

    findings = scan_path(tmp_path)

    assert any(
        finding.path == filename and finding.line_number == 1 and finding.kind == expected_kind
        for finding in findings
    )


def test_cli_reports_only_location_and_kind_without_secret_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "sk-" + "Z" * 32
    _write(tmp_path / "nested" / "settings.py", f'api_key = "{secret}"\n')

    exit_code = main([str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "nested/settings.py:1: DEEPSEEK_KEY" in captured.out
    assert secret not in captured.out
    assert secret not in captured.err


def test_cli_returns_success_for_clean_tree(tmp_path: Path) -> None:
    _write(tmp_path / "settings.example", 'token = "placeholder-only"\n')

    assert main([str(tmp_path)]) == 0
