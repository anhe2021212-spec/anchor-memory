from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if not match:
        raise ValueError(f"invalid requirement: {requirement}")
    return match.group(0)


def declared_requirements(root: Path) -> list[str]:
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    return sorted({requirement_name(item) for item in requirements}, key=str.casefold)


def distribution_license(metadata) -> str:
    direct = metadata.get("License-Expression") or metadata.get("License") or ""
    compact = " ".join(direct.split())
    if (
        compact
        and compact.casefold() != "unknown"
        and len(compact) <= 160
        and "http" not in compact.casefold()
    ):
        return compact
    classifiers = [
        item.split("::")[-1].strip()
        for item in metadata.get_all("Classifier", [])
        if item.startswith("License ::")
    ]
    return " / ".join(classifiers) or "unknown"


def component(name: str) -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "type": "library",
            "name": name,
            "version": "not-installed",
            "scope": "optional",
        }
    metadata = distribution.metadata
    license_name = distribution_license(metadata)
    return {
        "type": "library",
        "name": metadata.get("Name", name),
        "version": distribution.version,
        "licenses": [{"license": {"name": license_name}}],
        "purl": f"pkg:pypi/{metadata.get('Name', name).casefold()}@{distribution.version}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "anchor-memory",
                "version": project["version"],
            }
        },
        "components": [component(name) for name in declared_requirements(root)],
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(payload['components'])} SBOM components")


if __name__ == "__main__":
    main()
