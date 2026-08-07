from __future__ import annotations

import argparse

from .pipeline_cli import add_pipeline_arguments, config_from_args
from .real_data import prepare_real_manifest


def main() -> None:  # pragma: no cover - integration CLI
    parser = argparse.ArgumentParser(
        description="Download/index real ShareGPT or official LLaVA pretrain data"
    )
    add_pipeline_arguments(parser)
    config = config_from_args(parser.parse_args())
    prepare_real_manifest(config)


if __name__ == "__main__":
    main()
