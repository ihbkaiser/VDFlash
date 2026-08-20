#!/usr/bin/env python3
"""Normalize flat image-caption JSONL for standalone EAGLE3 Phase 2."""

from __future__ import annotations

import argparse

from specforge.data.qwen25vl_manifest import (
    materialize_image_archive,
    normalize_caption_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="flat source JSONL")
    parser.add_argument("--output", required=True, help="normalized manifest JSONL")
    parser.add_argument(
        "--image-root",
        default=None,
        help="directory containing images referenced by the source rows",
    )
    parser.add_argument(
        "--image-archive",
        default=None,
        help="optional zip/tar archive to materialize before validation",
    )
    parser.add_argument(
        "--materialized-image-root",
        default=None,
        help="controlled destination for --image-archive",
    )
    parser.add_argument("--expected-records", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_root = args.image_root
    if args.image_archive:
        image_root = args.materialized_image_root or image_root
        if not image_root:
            raise ValueError(
                "--image-archive requires --image-root or "
                "--materialized-image-root"
            )
        materialize_image_archive(args.image_archive, image_root)
    normalize_caption_jsonl(
        args.input,
        args.output,
        expected_records=args.expected_records,
        image_root=image_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
