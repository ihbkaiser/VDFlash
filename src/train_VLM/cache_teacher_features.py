from __future__ import annotations

import argparse

from .pipeline_cli import add_pipeline_arguments, config_from_args
from .teacher_cache import cache_teacher_features


def main() -> None:  # pragma: no cover - integration CLI
    parser = argparse.ArgumentParser(
        description="Cache frozen Qwen2.5-VL labels and selected hidden features into disk shards"
    )
    add_pipeline_arguments(parser)
    config = config_from_args(parser.parse_args())
    cache_teacher_features(config)


if __name__ == "__main__":
    main()
