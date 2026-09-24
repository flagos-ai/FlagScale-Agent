# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
"""Knowledge Manager for FlagScale Agent.

Manages knowledge groups: loading config, reading indexes, returning
doc content for the LLM to consume.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


class KnowledgeManager:
    """Manages knowledge index and document retrieval."""

    def __init__(self, knowledge_root: Optional[str] = None):
        if knowledge_root is None:
            knowledge_root = str(Path(__file__).parent)
        self.root = Path(knowledge_root)
        self.config_path = self.root / "knowledge_config.yaml"
        self.docs_path = self.root / "docs"
        self.indexes_path = self.root / "indexes"
        self._config: dict = {}
        self._load_config()

    def _load_config(self):
        """Load knowledge_config.yaml."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

    def list_groups(self) -> list[dict]:
        """List all available knowledge groups with descriptions."""
        groups = []
        for name, cfg in self._config.items():
            if name.startswith("_"):
                continue  # Skip metadata keys
            groups.append({
                "name": name,
                "description": cfg.get("description", ""),
                "doc_count": len(cfg.get("docs", [])),
            })
        return groups

    def get_index(self, group_name: str) -> Optional[str]:
        """Get the index content for a knowledge group."""
        idx_file = self.indexes_path / f"{group_name}.idx"
        if idx_file.exists():
            return idx_file.read_text(encoding="utf-8")
        return None

    def get_doc_content(
        self, doc_path: str, start_line: int = 1, end_line: Optional[int] = None
    ) -> Optional[str]:
        """Read doc content by relative path (within docs/)."""
        full_path = self.docs_path / doc_path
        if not full_path.exists():
            return None
        lines = full_path.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        start = max(1, start_line) - 1  # 0-based
        end = min(total, end_line) if end_line else total
        selected = lines[start:end]
        return "\n".join(selected)

    def get_group_docs(self, group_name: str) -> list[str]:
        """Get list of doc paths for a group."""
        cfg = self._config.get(group_name, {})
        return cfg.get("docs", [])

    def load_knowledge(self, group_name: str) -> Optional[str]:
        """Load full knowledge for a group: index + all docs content.

        Returns concatenated content of the index file followed by
        all document contents. For large groups, caller may want to
        use get_index() first and then selectively load docs.
        """
        if group_name not in self._config:
            return None

        parts = []
        # Add index as table of contents
        index = self.get_index(group_name)
        if index:
            parts.append(f"=== INDEX for {group_name} ===\n{index}\n")

        # Add all doc contents
        for doc_path in self.get_group_docs(group_name):
            content = self.get_doc_content(doc_path)
            if content:
                parts.append(f"=== {doc_path} ===\n{content}\n")

        return "\n".join(parts) if parts else None

    @property
    def available_groups(self) -> list[str]:
        """Return list of group names."""
        return [k for k in self._config.keys() if not k.startswith("_")]
