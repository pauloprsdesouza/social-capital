"""Dataset inventory for canonical raw CSVs under recsocial_py/data/raw/."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]
RAW = PKG / "data" / "raw"
DOCS_OUT = PKG.parent / "docs" / "DATA_INVENTORY.md"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def analyze_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    info: dict = {
        "path": str(path.relative_to(PKG.parent)),
        "rows": len(df),
        "columns": list(df.columns),
        "sha256_prefix": file_sha256(path),
    }
    for col in ("algorithm", "Algorithm"):
        if col in df.columns:
            info["algorithms"] = sorted(df[col].dropna().unique().tolist())
    for col in ("user_id", "ID_User"):
        if col in df.columns:
            info["unique_users"] = int(df[col].nunique())
    return info


def check_overlap() -> dict:
    ratings = pd.read_csv(RAW / "ratings.csv")
    tweets = pd.read_csv(RAW / "tweets.csv")
    rated_ids = set(ratings["item_id"].astype(str))
    tweet_ids = set(tweets["Id"].astype(str))
    return {
        "rated_items": len(rated_ids),
        "tweet_items": len(tweet_ids),
        "overlap": len(rated_ids & tweet_ids),
        "rated_missing_from_tweets": len(rated_ids - tweet_ids),
    }


def main() -> None:
    datasets = {
        "ratings": RAW / "ratings.csv",
        "tweets": RAW / "tweets.csv",
        "users_twitter": RAW / "users_twitter.csv",
        "legacy_v2_recommendations": RAW / "legacy" / "v2_recommendations.csv",
        "legacy_v3_output": RAW / "legacy" / "v3_output.csv",
    }
    entries = []
    for _, path in datasets.items():
        if not path.exists():
            entries.append({"path": str(path.relative_to(PKG.parent)), "missing": True})
            continue
        entries.append(analyze_csv(path))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Data Inventory",
        "",
        f"Auto-generated on {ts}.",
        "",
        "## Overlap (ratings ↔ tweets)",
        "",
        "```json",
        json.dumps(check_overlap(), indent=2),
        "```",
        "",
        "## Files",
        "",
    ]
    for e in entries:
        lines.append(f"### `{e.get('path', '?')}`")
        lines.append("")
        if e.get("missing"):
            lines.append("- **Missing**")
        else:
            lines.append(f"- Rows: **{e['rows']}**")
            lines.append(f"- SHA-256 prefix: `{e['sha256_prefix']}`")
            if "unique_users" in e:
                lines.append(f"- Unique users: **{e['unique_users']}**")
        lines.append("")

    DOCS_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DOCS_OUT}")


if __name__ == "__main__":
    main()
