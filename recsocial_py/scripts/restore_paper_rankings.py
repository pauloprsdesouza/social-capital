"""Restore author paper ranking CSVs from git history into data/raw/paper_rankings/."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "paper_rankings"

SOURCES = {
    "v2_recommendations.csv": "6f4ddf6:recsocial_py/data/raw/legacy/v2_recommendations.csv",
    "v2_baseline_recommendations.csv": "6f4ddf6:legacy/outputs/updated_recommendations.csv",
    "v3_recommendations.csv": "6f4ddf6:legacy/outputs/output_v3.csv",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, git_path in SOURCES.items():
        raw = subprocess.check_output(
            ["git", "show", git_path],
            cwd=ROOT.parent,
            text=True,
        )
        (OUT / name).write_text(raw, encoding="utf-8")
        print(f"Wrote {name} ({len(raw.splitlines())} lines)")


if __name__ == "__main__":
    main()
