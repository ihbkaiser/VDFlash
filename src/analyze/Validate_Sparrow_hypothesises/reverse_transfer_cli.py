"""CLI entry point for the trained-MSD reverse-transfer experiment."""

from __future__ import annotations

import argparse

from .run_reverse_transfer import build_parser as _build_run_parser, run


def build_parser() -> argparse.ArgumentParser:
    return _build_run_parser()


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
