#!/bin/sh
set -e
# Runs the make target from RUN_TARGET (default: dag_math_flush_per_workflow).
# Set RUN_TARGET when starting the stack, e.g. RUN_TARGET=dag_math_flush_per_request docker compose up.
mkdir -p "$(dirname "$OUTPUT_PATH")"
TARGET="${RUN_TARGET:-dag_math_flush_per_workflow}"
echo "exec make $TARGET"
exec make "$TARGET"
