#!/usr/bin/env python3
"""Run V3 pipeline (delegates to recsocial CLI)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main() -> int:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(PKG / "src")}
    cmd = [sys.executable, "-m", "recsocial.cli", "run", "v3"]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PKG), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
