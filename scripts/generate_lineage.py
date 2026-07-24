#!/usr/bin/env python3
"""Generate public structural lineage without exposing origin paths."""
from __future__ import annotations

import argparse
import ast
import hashlib
from collections import Counter
from pathlib import Path


HEADER = (
    "origin_file\torigin_sha256\trelease_sha256\torigin_lines\trelease_lines\t"
    "origin_symbols\tretained_symbols\texact_code_line_retention"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbols(path: Path) -> Counter[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return Counter(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )


def code_lines(path: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            result[line] += 1
    return result


def release_by_name(root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in parts):
            continue
        if any(
            part in {"build", "dist", "__pycache__"} or part.endswith(".egg-info")
            for part in parts
        ):
            continue
        candidates.setdefault(path.name, []).append(path)
    duplicates = {name: paths for name, paths in candidates.items() if len(paths) > 1}
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise RuntimeError(f"ambiguous release basenames: {names}")
    return {name: paths[0] for name, paths in candidates.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-dir", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("docs/lineage.tsv"))
    args = parser.parse_args()
    origin_dir = args.origin_dir.resolve()
    release_root = args.release_root.resolve()
    release = release_by_name(release_root)
    rows = [HEADER]
    for origin in sorted(origin_dir.glob("*.py"), key=lambda path: path.name):
        target = release.get(origin.name)
        if target is None:
            raise RuntimeError(f"allowlisted origin has no release target: {origin.name}")
        origin_symbols = symbols(origin)
        release_symbols = symbols(target)
        retained = sum((origin_symbols & release_symbols).values())
        origin_code = code_lines(origin)
        release_code = code_lines(target)
        exact = sum((origin_code & release_code).values())
        denominator = sum(origin_code.values()) or 1
        rows.append(
            "\t".join(
                [
                    origin.name,
                    digest(origin),
                    digest(target),
                    str(len(origin.read_text(encoding="utf-8").splitlines())),
                    str(len(target.read_text(encoding="utf-8").splitlines())),
                    str(sum(origin_symbols.values())),
                    str(retained),
                    f"{exact / denominator:.3f}",
                ]
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
