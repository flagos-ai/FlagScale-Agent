#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
"""Generate knowledge index files from config and docs.

Scans each knowledge group in knowledge_config.yaml, extracts section headers
(## / ###) with line numbers from each doc, and writes a compact index file
per group into indexes/ directory.

Usage:
    python generate_index.py              # uses default paths
    python generate_index.py --config /path/to/config.yaml --docs /path/to/docs --output /path/to/indexes
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml


def extract_sections(filepath: str) -> list[dict]:
    """Extract markdown section headers with line numbers from a file."""
    sections = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                match = re.match(r"^(#{1,4})\s+(.+)", line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    sections.append(
                        {"level": level, "title": title, "line": line_num}
                    )
    except FileNotFoundError:
        print(f"  WARNING: file not found: {filepath}", file=sys.stderr)
    return sections


def count_lines(filepath: str) -> int:
    """Count total lines in a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def generate_index_for_group(
    group_name: str, group_config: dict, docs_root: str, sources: dict = None
) -> str:
    """Generate index content for a single knowledge group."""
    lines = []
    lines.append(f"# {group_name}")
    lines.append(f"# {group_config.get('description', '')}")

    # Add source/commit info if available
    if sources:
        doc_paths = group_config.get("docs", [])
        if doc_paths:
            # Infer source repo from first doc's directory name
            first_dir = doc_paths[0].split("/")[0] if "/" in doc_paths[0] else ""
            if first_dir and first_dir in sources:
                src = sources[first_dir]
                lines.append(f"# Source: {src.get('repo', first_dir)} @ {src.get('commit', 'unknown')}")

    lines.append("")
    lines.append(
        "# To read a section: load_knowledge(name='"
        f"{group_name}', doc='<path below>', start_line=<L>, end_line=<L>)"
    )
    lines.append("# Do NOT pass these doc paths to read_file — they are knowledge-internal.")
    lines.append("")

    doc_paths = group_config.get("docs", [])
    for doc_rel in doc_paths:
        doc_path = os.path.join(docs_root, doc_rel)
        total_lines = count_lines(doc_path)
        sections = extract_sections(doc_path)

        lines.append(f"## {doc_rel} ({total_lines} lines)")
        for sec in sections:
            indent = "  " * (sec["level"] - 1)
            lines.append(f"{indent}L{sec['line']}: {sec['title']}")
        lines.append("")

    return "\n".join(lines)


def collect_group_stats(config: dict, docs_root: str) -> list[dict]:
    """Collect per-group statistics: doc count, line counts, averages."""
    stats = []
    for group_name, group_config in config.items():
        if group_name.startswith("_"):
            continue
        doc_paths = group_config.get("docs", [])
        doc_stats = []
        for doc_rel in doc_paths:
            doc_path = os.path.join(docs_root, doc_rel)
            lines = count_lines(doc_path)
            # Extract chapter title from first H1
            title = doc_rel.split("/")[-1].replace(".md", "")
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    for line in f:
                        m = re.match(r"^#\s+(.+)", line)
                        if m:
                            title = m.group(1).strip()
                            # Remove "第N章：" prefix for brevity
                            title = re.sub(r"^第\d+章[：:]\s*", "", title)
                            break
            except FileNotFoundError:
                pass
            doc_stats.append({"path": doc_rel, "title": title, "lines": lines})
        
        total_lines = sum(d["lines"] for d in doc_stats)
        avg_lines = total_lines // len(doc_stats) if doc_stats else 0
        line_range = f"{min(d['lines'] for d in doc_stats)}-{max(d['lines'] for d in doc_stats)}" if doc_stats else "0"
        
        stats.append({
            "group": group_name,
            "description": group_config.get("description", ""),
            "doc_count": len(doc_stats),
            "docs": doc_stats,
            "total_lines": total_lines,
            "avg_lines": avg_lines,
            "line_range": line_range,
        })
    return stats


def update_quality_standard(docs_root: str, all_stats: list[dict]):
    """Auto-update the stats section in 01_infrastructure_analysis.md."""
    target_file = os.path.join(docs_root, "analysis_standards", "01_infrastructure_analysis.md")
    if not os.path.exists(target_file):
        print(f"  [skip] quality standard file not found: {target_file}")
        return

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the marker section "## 六、示例产出指标" and replace everything after it
    marker = "## 六、示例产出指标"
    marker_pos = content.find(marker)
    if marker_pos == -1:
        print(f"  [skip] marker '{marker}' not found in quality standard")
        return

    # Build new stats section
    total_docs = sum(s["doc_count"] for s in all_stats)
    total_lines = sum(s["total_lines"] for s in all_stats)
    
    new_section = f"{marker}\n\n"
    new_section += f"基于知识库已有的 **{total_docs} 篇文档**实践数据"
    new_section += f"（覆盖 {len(all_stats)} 个知识组，总计 ~{total_lines:,} 行）：\n\n"

    # Group stats into "detailed" (≥5 docs) and "summary" (< 5 docs)
    detailed_groups = [s for s in all_stats if s["doc_count"] >= 5]
    summary_groups = [s for s in all_stats if s["doc_count"] < 5]

    for gs in sorted(detailed_groups, key=lambda x: -x["total_lines"]):
        # Use short display name from group description
        desc = gs["description"].split("：")[0] if "：" in gs["description"] else gs["group"]
        new_section += f"### {gs['group']}（{gs['doc_count']}章，平均 ~{gs['avg_lines']} 行）\n\n"
        new_section += "| 章节 | 主题 | 行数 |\n"
        new_section += "|------|------|------|\n"
        for doc in gs["docs"]:
            fname = doc["path"].split("/")[-1].replace(".md", "")
            new_section += f"| {fname} | {doc['title']} | {doc['lines']} |\n"
        new_section += "\n"

    if summary_groups:
        new_section += "### 其他知识组汇总\n\n"
        new_section += "| 知识组 | 章数 | 平均行数 | 行数范围 |\n"
        new_section += "|--------|------|----------|----------|\n"
        for gs in sorted(summary_groups, key=lambda x: -x["avg_lines"]):
            desc = gs["description"].split("：")[0] if "：" in gs["description"] else gs["group"]
            new_section += f"| {desc} | {gs['doc_count']} | ~{gs['avg_lines']} | {gs['line_range']} |\n"
        new_section += "\n"

    new_section += "### 总结\n\n"
    new_section += f"- 全库 {total_docs} 篇文档，总计 **~{total_lines:,} 行**\n"
    
    # Calculate core vs topic averages
    core_groups = [s for s in all_stats if s["avg_lines"] >= 400]
    topic_groups = [s for s in all_stats if s["avg_lines"] < 400]
    if core_groups:
        core_avg = sum(s["avg_lines"] for s in core_groups) // len(core_groups)
        new_section += f"- 核心深度分析章节：平均 **{core_avg} 行**\n"
    if topic_groups:
        topic_avg = sum(s["avg_lines"] for s in topic_groups) // len(topic_groups)
        new_section += f"- 专题分析：平均 **{topic_avg} 行**\n"
    new_section += "- 推荐标准：核心模块 ≥ 450 行，专题/工具类 ≥ 250 行\n"

    # Replace content
    new_content = content[:marker_pos] + new_section
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    new_line_count = new_content.count("\n") + 1
    print(f"  [updated] {target_file} ({new_line_count} lines)")


def main():
    # Determine base directory
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Generate knowledge indexes")
    parser.add_argument(
        "--config",
        default=str(base_dir / "knowledge_config.yaml"),
        help="Path to knowledge_config.yaml",
    )
    parser.add_argument(
        "--docs",
        default=str(base_dir / "docs"),
        help="Path to docs directory",
    )
    parser.add_argument(
        "--output",
        default=str(base_dir / "indexes"),
        help="Path to output indexes directory",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip updating quality standard stats",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    total_groups = 0
    total_docs = 0

    # Extract _sources metadata (skip from group iteration)
    sources = config.pop("_sources", {}) or {}

    for group_name, group_config in config.items():
        if group_name.startswith("_"):
            continue  # Skip metadata keys
        index_content = generate_index_for_group(
            group_name, group_config, args.docs, sources=sources
        )
        output_file = os.path.join(args.output, f"{group_name}.idx")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(index_content)

        doc_count = len(group_config.get("docs", []))
        total_groups += 1
        total_docs += doc_count
        print(f"  [{group_name}] {doc_count} docs -> {output_file}")

    print(f"\nDone: {total_groups} indexes, {total_docs} docs indexed.")

    # Auto-update quality standard stats
    if not args.no_stats:
        print("\nUpdating quality standard stats...")
        # Reload config (since we popped _sources)
        with open(args.config, "r", encoding="utf-8") as f:
            config_fresh = yaml.safe_load(f)
        all_stats = collect_group_stats(config_fresh, args.docs)
        update_quality_standard(args.docs, all_stats)


if __name__ == "__main__":
    main()
