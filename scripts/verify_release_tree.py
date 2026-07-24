from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

SKIP_NAMES = {".git"}
GENERATED_NAMES = {"build", "dist"}
FORBIDDEN_NAMES = {
    ".env",
    ".secrets",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "logs",
    "drafts",
    "mail_drafts",
    "review",
}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".wal",
    ".shm",
    ".bak",
    ".orig",
    ".log",
    ".pyc",
}
TEXT_SUFFIXES = {
    "",
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yml",
    ".yaml",
    ".ini",
    ".conf",
    ".example",
    ".service",
    ".timer",
}
BINARY_HEADERS = {
    b"SQLite format 3\x00": "sqlite-header",
    b"PAR1": "parquet-header",
}
ALLOWED_PUBLIC_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "api.voyageai.com", "api.anthropic.com", "api.openai.com", "api.deepseek.com",
    "github.com",
}
ALLOWED_PUBLIC_URLS = (
    "https:" + "//github.com/limen-threshold/anchor-memory/commit/"
    "8dd21337c2b87ff6d876dc269b475242650ea56e",
    "https:" + "//github.com/limen-threshold/anchor-memory",
)


def _rules() -> dict[str, re.Pattern[str]]:
    private_unix = "/" + "(?:root|home/[A-Za-z0-9._-]+)" + "/"
    private_windows = r"[A-Za-z]:\\Users\\" + r"[^\\\s]+\\"
    coordinate_decimal = r"(?<!\d)-?\d{1,3}\.\d{4,}(?!\d)"
    return {
        "private-unix-path": re.compile(private_unix),
        "private-windows-path": re.compile(private_windows, re.IGNORECASE),
        "ipv4-address": re.compile(
            r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
        ),
        "email-address": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
        ),
        "hardcoded-bearer": re.compile(
            r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE
        ),
        "hardcoded-basic": re.compile(
            r"\bBasic\s+[A-Za-z0-9+/=]{12,}", re.IGNORECASE
        ),
        "provider-key-shape": re.compile(
            r"\b(?:sk|rk|pk|api)[-_][A-Za-z0-9_-]{16,}\b", re.IGNORECASE
        ),
        "private-identity-literal": re.compile(
            "|".join(("安" + "珩", "安" + "老师", "C" + "老师", "小" + "狗")),
            re.IGNORECASE,
        ),
        "private-channel-literal": re.compile("an" + r"\s+channel", re.IGNORECASE),
        "phone-shape": re.compile(r"(?<!\d)\+?\d[\d ()-]{9,}\d(?!\d)"),
        "coordinate-shape": re.compile(
            coordinate_decimal + r",\s*" + coordinate_decimal
        ),
    }


def entropy(value: str) -> float:
    counts = {character: value.count(character) for character in set(value)}
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def scan(root: Path) -> list[tuple[str, Path, int | None]]:
    findings: list[tuple[str, Path, int | None]] = []
    rules = _rules()
    url_pattern = re.compile("http" + r"s?://[^\s\"'<>]+", re.IGNORECASE)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_NAMES for part in relative.parts):
            continue
        if any(
            part in GENERATED_NAMES or part.endswith(".egg-info")
            for part in relative.parts
        ):
            findings.append(("generated-artifact", relative, None))
            continue
        if path.is_symlink():
            findings.append(("symlink", relative, None))
            continue
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            findings.append(("forbidden-name", relative, None))
            continue
        if path.is_dir():
            continue
        lower_name = path.name.casefold()
        if (
            path.suffix.casefold() in FORBIDDEN_SUFFIXES
            or "-wal" in lower_name
            or "-shm" in lower_name
            or ".bak-" in lower_name
        ):
            findings.append(("forbidden-suffix", relative, None))
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(("oversized-file", relative, None))
            continue
        prefix = path.read_bytes()[:32]
        for header, rule in BINARY_HEADERS.items():
            if prefix.startswith(header):
                findings.append((rule, relative, None))
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append(("unknown-binary", relative, None))
                continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.name == "manifest.sha256":
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            sanitized_line = line.replace("YOUR_API_KEY_HERE", "")
            if path.name == "lineage.tsv":
                sanitized_line = re.sub(r"\b[0-9a-f]{64}\b", "", sanitized_line)
            for allowed_url in ALLOWED_PUBLIC_URLS:
                sanitized_line = sanitized_line.replace(allowed_url, "")
            rule_line = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", "", sanitized_line)
            for rule, pattern in rules.items():
                match = pattern.search(rule_line)
                if not match:
                    continue
                if rule == "ipv4-address" and match.group(0) in {"127.0.0.1", "0.0.0.0"}:
                    continue
                if rule == "phone-shape" and sum(ch.isdigit() for ch in match.group(0)) < 10:
                    continue
                findings.append((rule, relative, line_number))
            for url in url_pattern.findall(sanitized_line):
                host = url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
                if host not in ALLOWED_PUBLIC_HOSTS and not host.endswith(".invalid"):
                    findings.append(("public-url", relative, line_number))
            for candidate in re.findall(
                r"(?<![A-Za-z0-9])[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9])",
                sanitized_line,
            ):
                edge_roles = {"lateral", "temporal", "derived_from", "updates", "SUPPORTED_BY", "EVOKES"}
                if set(candidate.split("/")) == edge_roles:
                    continue
                if entropy(candidate) >= 4.4:
                    findings.append(("high-entropy-string", relative, line_number))
                    break
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings = scan(root)
    for rule, path, line in findings:
        location = f"{path}:{line}" if line else str(path)
        print(f"{rule}\t{location}")
    if findings:
        print(f"release verification failed: {len(findings)} finding(s)")
        raise SystemExit(1)
    print("release verification passed")


if __name__ == "__main__":
    main()
