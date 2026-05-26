#!/usr/bin/env python3
"""Audit decision archive ordering and link safety before any mass rename."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ORDERED_PREFIX_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|000\d)-")
DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")
DECISION_LINK_RE = re.compile(
    r"(?P<target>(?:docs/wiki/)?decisions/(?P<filename>[A-Za-z0-9_.-]+\.(?:md|json|jsonl)))"
    r"(?P<anchor>#[A-Za-z0-9_.~:%/?=&-]+)?"
)
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".rst",
}


def is_ordered_filename(name: str) -> bool:
    return bool(ORDERED_PREFIX_RE.match(name))


def suffix_for_name(name: str) -> str:
    path = Path(name)
    return "".join(path.suffixes) if path.suffixes else ""


def stem_without_suffixes(name: str) -> str:
    suffix = suffix_for_name(name)
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(name).stem


def normalize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "decision"


def proposed_ordered_name(name: str) -> str | None:
    if is_ordered_filename(name):
        return None
    suffix = suffix_for_name(name)
    stem = stem_without_suffixes(name)
    match = DATE_RE.search(stem)
    if match:
        date = match.group("date")
        before = stem[: match.start("date")]
        after = stem[match.end("date") :]
        slug = normalize_slug(f"{before}-{after}")
        return f"{date}-{slug}{suffix}"
    return f"archive-unknown-{name}"


def text_files(scan_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        if scan_dir.is_file():
            if scan_dir.suffix in TEXT_SUFFIXES:
                files.append(scan_dir)
            continue
        for path in scan_dir.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def audit_links(
    *,
    root: Path,
    decisions_dir: Path,
    scan_dirs: list[Path],
) -> list[dict[str, Any]]:
    existing = {path.name for path in decisions_dir.iterdir() if path.is_file()} if decisions_dir.exists() else set()
    broken: list[dict[str, Any]] = []
    for path in text_files(scan_dirs):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in DECISION_LINK_RE.finditer(line):
                filename = match.group("filename")
                if filename not in existing:
                    broken.append(
                        {
                            "source": relative_to_root(path, root),
                            "line": int(line_no),
                            "target": match.group("target"),
                            "filename": filename,
                        }
                    )
    return broken


def collision_report(rename_candidates: dict[str, str]) -> dict[str, list[str]]:
    by_target: dict[str, list[str]] = {}
    for old, new in rename_candidates.items():
        by_target.setdefault(new, []).append(old)
    return {target: olds for target, olds in sorted(by_target.items()) if len(olds) > 1}


def audit_decision_archive(
    *,
    root: Path,
    decisions_dir: Path,
    scan_dirs: list[Path],
) -> dict[str, Any]:
    decision_files = sorted(path.name for path in decisions_dir.iterdir() if path.is_file())
    unordered = [name for name in decision_files if not is_ordered_filename(name)]
    rename_candidates = {
        name: proposed
        for name in unordered
        if (proposed := proposed_ordered_name(name)) is not None
    }
    broken_links = audit_links(root=root, decisions_dir=decisions_dir, scan_dirs=scan_dirs)
    report = {
        "decision": "decision_archive_ordering_audit",
        "root": str(root),
        "decisions_dir": str(decisions_dir),
        "decision_file_count": int(len(decision_files)),
        "ordered_file_count": int(len(decision_files) - len(unordered)),
        "unordered_file_count": int(len(unordered)),
        "unordered_files": unordered,
        "rename_candidates": rename_candidates,
        "rename_collision_count": int(len(collision_report(rename_candidates))),
        "rename_collisions": collision_report(rename_candidates),
        "broken_link_count": int(len(broken_links)),
        "broken_links": broken_links,
        "plain_language_read": (
            "This is a dry-run guard for the decisions archive. It identifies "
            "which files still sort like old memory, proposes non-destructive "
            "ordered names, and finds links that would already be broken before "
            "any rename is applied."
        ),
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--decisions-dir", default="docs/wiki/decisions")
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=[],
        help="Directory or file to scan for decision links. Can be passed multiple times.",
    )
    parser.add_argument("--out-json", default="")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    decisions_dir = Path(args.decisions_dir)
    if not decisions_dir.is_absolute():
        decisions_dir = root / decisions_dir
    scan_dirs = [Path(item) for item in args.scan_dir]
    if not scan_dirs:
        scan_dirs = [root / "docs", root / "scripts", root / "tests", root / "src"]
    scan_dirs = [path if path.is_absolute() else root / path for path in scan_dirs]
    report = audit_decision_archive(
        root=root,
        decisions_dir=decisions_dir,
        scan_dirs=scan_dirs,
    )
    if str(args.out_json):
        out = Path(args.out_json)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
