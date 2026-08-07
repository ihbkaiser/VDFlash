from __future__ import annotations

import argparse

from .cached_trainer import train_cached_draft
from .pipeline_cli import add_pipeline_arguments, config_from_args


def main() -> None:  # pragma: no cover - integration CLI
    parser = argparse.ArgumentParser(
        description="Train vanilla DFlash from offline real-data teacher feature shards"
    )
    add_pipeline_arguments(parser)
    config = config_from_args(parser.parse_args())
    train_cached_draft(config)


if __name__ == "__main__":
    main()
