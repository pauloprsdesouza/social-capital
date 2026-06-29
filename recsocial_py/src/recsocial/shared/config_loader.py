"""YAML config loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml_config(
    path: str | Path,
    model: type[T],
    base_dir: Path | None = None,
    *,
    resolve_fn: Callable[[T, Path], T] | None = None,
) -> T:
    path = Path(path)
    base = base_dir or path.parent.parent
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    cfg = model.model_validate(raw)
    return resolve_fn(cfg, base) if resolve_fn else cfg


def resolve_paths_in_dict(
    paths: dict[str, str],
    base_dir: Path,
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    resolved = dict(paths)
    for key in keys or paths.keys():
        val = resolved.get(key)
        if val is None:
            continue
        p = Path(val)
        if not p.is_absolute():
            resolved[key] = str((base_dir / p).resolve())
    return resolved
