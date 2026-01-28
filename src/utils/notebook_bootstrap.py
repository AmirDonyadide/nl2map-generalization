# src/utils/notebook_bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None, marker_dir: str = "src") -> Path:
    """
    Walk upwards from `start` (default: cwd) until we find a directory that contains `marker_dir`.
    Returns the repo root Path.

    Raises RuntimeError if not found.
    """
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / marker_dir).is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root (no '{marker_dir}' folder found in parents of {p}).")


def add_repo_root_to_syspath(start: Optional[Path] = None, marker_dir: str = "src") -> Path:
    """
    Find repo root (folder containing `marker_dir`, default: 'src') and prepend it to sys.path.

    Returns the repo root.
    """
    root = find_repo_root(start=start, marker_dir=marker_dir)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
