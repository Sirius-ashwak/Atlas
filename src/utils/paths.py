"""Utility helpers for dealing with filesystem paths."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root directory.

    The function resolves the location of this utility module inside ``src`` and
    climbs two levels up the directory hierarchy which corresponds to the
    repository root when the project follows the standard layout used here.
    It provides a single reliable way for CLI scripts located in
    ``python_scripts/`` to reference shared resources such as the ``configs/``
    and ``models/`` folders.
    """

    return Path(__file__).resolve().parents[2]


__all__ = ["get_project_root"]
