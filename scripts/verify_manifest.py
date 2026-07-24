from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = root / "manifest.sha256"
    if args.write:
        rows = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == manifest:
                continue
            relative = path.relative_to(root)
            if ".git" in relative.parts:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {relative.as_posix()}")
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} manifest entries")
        return
    failures = []
    expected_files = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        expected_files.add(relative)
        path = root / relative
        if not path.is_file():
            failures.append((relative, "missing"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append((relative, "digest-mismatch"))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.sha256"
        and ".git" not in path.relative_to(root).parts
    }
    for relative in sorted(actual_files - expected_files):
        failures.append((relative, "unexpected"))
    for relative in sorted(expected_files - actual_files):
        if not (root / relative).is_file():
            failures.append((relative, "missing-from-tree"))
    for relative, rule in failures:
        print(f"{rule}\t{relative}")
    if failures:
        raise SystemExit(1)
    print("manifest verification passed")


if __name__ == "__main__":
    main()
