"""Figure gallery index for report directories."""

from __future__ import annotations

from pathlib import Path


def write_figures_index(figures_dir: Path, entries: list[tuple[str, str]]) -> Path:
    """Write figures/index.md listing generated charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure Gallery",
        "",
        "Auto-generated charts for paper-style validation.",
        "",
    ]
    for filename, caption in entries:
        lines.extend([f"## {caption}", "", f"![{caption}]({filename})", ""])
    index = figures_dir / "index.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index
