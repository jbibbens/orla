#!/bin/sh
set -e
# Runs the given make target (e.g. baseline). Pass the target as the container command:
#   docker compose run --rm run baseline
mkdir -p "$(dirname "$OUTPUT_PATH")"
exec make "$@"
