"""Repository-relative paths shared by inference entry points."""

from __future__ import annotations

import os
from pathlib import Path


def _workspace_candidate(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _looks_like_workspace(candidate: Path) -> bool:
    return candidate.is_dir() and candidate.joinpath("src").is_dir() and (
        candidate.joinpath(".git").exists()
        or candidate.joinpath("externals").is_dir()
    )


def find_workspace_root(start: str | Path | None = None) -> Path:
    """Find the checkout root without depending on the caller's ``cwd``.

    ``VD_FLASH_WORKSPACE`` is an explicit override for mounted/shared
    workspaces. Otherwise the function walks upward from ``start`` (or this
    module) until it finds the repository markers used by this checkout.
    """

    override = os.environ.get("VD_FLASH_WORKSPACE")
    if override:
        root = _workspace_candidate(override)
        if _looks_like_workspace(root):
            return root
        raise RuntimeError(
            "VD_FLASH_WORKSPACE does not point to a VDFlash checkout: "
            f"{root}"
        )

    origin = _workspace_candidate(start or __file__)
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if _looks_like_workspace(candidate):
            return candidate
    raise RuntimeError(
        "Could not find VDFlash workspace from: "
        f"{origin}. Set VD_FLASH_WORKSPACE to the checkout root."
    )


WORKSPACE_ROOT = find_workspace_root()


def workspace_path(*parts: str | os.PathLike[str]) -> Path:
    """Return an absolute path rooted at the detected checkout."""

    return WORKSPACE_ROOT.joinpath(*parts)


def resolve_workspace_path(value: str | os.PathLike[str]) -> Path:
    """Resolve relative CLI paths against the checkout instead of ``cwd``."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE_ROOT / path).resolve()


def resolve_namespace_paths(namespace: object, *names: str) -> object:
    """Resolve selected path-like argparse fields in place and return them."""

    for name in names:
        value = getattr(namespace, name, None)
        if value is not None:
            setattr(namespace, name, resolve_workspace_path(value))
    return namespace


def workspace_python(root: str | Path | None = None) -> Path:
    """Return the only supported project interpreter, ``<root>/.venv``."""

    root_path = _workspace_candidate(root) if root is not None else WORKSPACE_ROOT
    return root_path / ".venv" / "bin" / "python"


__all__ = [
    "WORKSPACE_ROOT",
    "find_workspace_root",
    "resolve_namespace_paths",
    "resolve_workspace_path",
    "workspace_path",
    "workspace_python",
]
