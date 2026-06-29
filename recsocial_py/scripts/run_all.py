#!/usr/bin/env python3
"""Run one or all Social Capital reproduction pipelines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main() -> int:
    slice_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    validate = "--validate" in sys.argv or slice_name == "all"
    cmd = [sys.executable, "-m", "recsocial.cli", "run", slice_name]
    if validate and slice_name != "all":
        cmd.append("--validate")
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(PKG / "src")}
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PKG), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
