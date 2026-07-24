from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _platform_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "anchor-memory"


def _clean_path(value: str | Path | None) -> Path | None:
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class AnchorConfig:
    data_dir: Path
    db_path: Path
    review_dir: Path
    log_dir: Path
    created_by: str = "local-user"
    uuid_namespace: str = "https://anchor-memory.example.invalid"
    embedding_provider: str = "local"
    embedding_url: str = "https://api.example.invalid/v1"
    embedding_api_key: str = ""
    taxonomy_url: str = "https://api.example.invalid/v1"
    taxonomy_api_key: str = ""
    taxonomy_model: str = ""
    chat_history_path: Path | None = None
    chroma_dir: Path | None = None
    kuzu_dir: Path | None = None
    recall_budget: int = 6
    flow_depth: int = 3
    flow_visit_multiplier: int = 5

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        review_dir: str | Path | None = None,
    ) -> "AnchorConfig":
        defaults = _platform_data_dir()
        values: dict[str, object] = {}
        path = _clean_path(config_file or os.environ.get("ANCHOR_CONFIG_FILE"))
        if path:
            with path.open("rb") as handle:
                values.update(tomllib.load(handle).get("anchor", {}))

        resolved_data = _clean_path(
            data_dir or os.environ.get("ANCHOR_DATA_DIR") or values.get("data_dir")
        ) or defaults
        resolved_db = _clean_path(
            db_path or os.environ.get("ANCHOR_DB_PATH") or values.get("db_path")
        ) or (resolved_data / "memories.db")
        resolved_review = _clean_path(
            review_dir
            or os.environ.get("ANCHOR_REVIEW_DIR")
            or values.get("review_dir")
        ) or (resolved_data / "review")
        resolved_log = _clean_path(
            os.environ.get("ANCHOR_LOG_DIR") or values.get("log_dir")
        ) or (resolved_data / "logs")

        cfg = cls(
            data_dir=resolved_data,
            db_path=resolved_db,
            review_dir=resolved_review,
            log_dir=resolved_log,
            created_by=str(os.environ.get("ANCHOR_CREATED_BY") or values.get("created_by") or "local-user"),
            uuid_namespace=str(values.get("uuid_namespace") or "https://anchor-memory.example.invalid"),
            embedding_provider=str(os.environ.get("ANCHOR_EMBEDDING_PROVIDER") or values.get("embedding_provider") or "local"),
            embedding_url=str(os.environ.get("ANCHOR_EMBEDDING_URL") or values.get("embedding_url") or "https://api.example.invalid/v1"),
            embedding_api_key=str(os.environ.get("ANCHOR_EMBEDDING_API_KEY") or ""),
            taxonomy_url=str(
                os.environ.get("ANCHOR_TAXONOMY_URL")
                or values.get("taxonomy_url")
                or "https://api.example.invalid/v1"
            ),
            taxonomy_api_key=str(os.environ.get("ANCHOR_TAXONOMY_API_KEY") or ""),
            taxonomy_model=str(
                os.environ.get("ANCHOR_TAXONOMY_MODEL")
                or values.get("taxonomy_model")
                or ""
            ),
            chat_history_path=_clean_path(
                os.environ.get("ANCHOR_CHAT_HISTORY_PATH")
                or values.get("chat_history_path")
            ),
            chroma_dir=_clean_path(
                os.environ.get("ANCHOR_CHROMA_DIR") or values.get("chroma_dir")
            ) or (resolved_data / "chroma"),
            kuzu_dir=_clean_path(
                os.environ.get("ANCHOR_KUZU_DIR") or values.get("kuzu_dir")
            ) or (resolved_data / "kuzu_db"),
            recall_budget=int(values.get("recall_budget", 6)),
            flow_depth=int(values.get("flow_depth", 3)),
            flow_visit_multiplier=int(values.get("flow_visit_multiplier", 5)),
        )
        return cfg.validate()

    def validate(self) -> "AnchorConfig":
        if self.recall_budget < 1 or self.flow_depth < 0:
            raise ValueError("recall_budget must be positive and flow_depth non-negative")
        placeholder = "YOUR_API_KEY_HERE"
        if self.embedding_api_key == placeholder:
            raise ValueError("placeholder credentials are not usable")
        if self.embedding_api_key and ".invalid" in self.embedding_url:
            raise ValueError("a real credential cannot be sent to an example.invalid URL")
        if self.taxonomy_api_key and ".invalid" in self.taxonomy_url:
            raise ValueError("a real credential cannot be sent to an example.invalid URL")
        if bool(self.taxonomy_api_key) != bool(self.taxonomy_model):
            raise ValueError("taxonomy_api_key and taxonomy_model must be configured together")
        return replace(
            self,
            data_dir=self.data_dir.resolve(),
            db_path=self.db_path.resolve(),
            review_dir=self.review_dir.resolve(),
            log_dir=self.log_dir.resolve(),
            chroma_dir=self.chroma_dir.resolve() if self.chroma_dir else None,
            kuzu_dir=self.kuzu_dir.resolve() if self.kuzu_dir else None,
            chat_history_path=(
                self.chat_history_path.resolve() if self.chat_history_path else None
            ),
        )

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.db_path.parent, self.review_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        if self.chroma_dir:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
        if self.kuzu_dir:
            self.kuzu_dir.parent.mkdir(parents=True, exist_ok=True)
