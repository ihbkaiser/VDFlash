#!/usr/bin/env bash
# Ready-to-run cache-only H1.1 profile for the supplied Phase 2 paths.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${H11_ENV_FILE:-"$SCRIPT_DIR/h11_3b_20e_cache.env"}
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing local H1.1 profile: $ENV_FILE" >&2
  echo "Copy scripts/h11.env.example to that path and fill in your Phase 2 paths." >&2
  exit 2
fi
exec bash "$SCRIPT_DIR/run_dflash_h11.sh" \
  --env-file "$ENV_FILE"
