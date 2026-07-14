"""Reject development-only learning markers before final submission."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKERS = ("TODO" + "(student)", "HI" + "NT:", "REF" + "ERENCE:")
SEARCH_DIRECTORIES = ("src", "app", "tests", "notebooks")
SEARCH_SUFFIXES = {".py", ".ipynb", ".md"}


def find_markers() -> list[str]:
    findings: list[str] = []
    for directory_name in SEARCH_DIRECTORIES:
        directory = PROJECT_ROOT / directory_name
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in SEARCH_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            for marker in MARKERS:
                if marker in content:
                    findings.append(f"{path.relative_to(PROJECT_ROOT)}: {marker}")
    return findings


def main() -> None:
    findings = find_markers()
    if findings:
        details = "\n".join(f"- {finding}" for finding in findings)
        raise SystemExit(f"Development markers remain:\n{details}")
    print("No development-only learning markers found.")


if __name__ == "__main__":
    main()

