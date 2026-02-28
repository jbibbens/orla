#!/bin/sh
set -e

# Usage: entrypoint.sh [path_to_instance.json] [baseline flags...]
# Reads instance JSON, clones repo at base_commit into WORKDIR_ROOT/<instance_id>, runs baseline.
# Extra args (e.g. -max-steps 10) are passed to the baseline binary.
# Requires: ORLA_URL, WORKDIR_ROOT, OUTPUT_PATH (env).

INSTANCE_PATH="${1:-/instances/instance.json}"
shift 2>/dev/null || true

if [ ! -f "$INSTANCE_PATH" ]; then
  echo "Instance file not found: $INSTANCE_PATH" >&2
  exit 1
fi

INSTANCE_ID=$(jq -r '.instance_id' "$INSTANCE_PATH")
REPO=$(jq -r '.repo' "$INSTANCE_PATH")
BASE_COMMIT=$(jq -r '.base_commit' "$INSTANCE_PATH")

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
  echo "Missing instance_id in $INSTANCE_PATH" >&2
  exit 1
fi
if [ -z "$REPO" ] || [ "$REPO" = "null" ]; then
  echo "Missing repo in $INSTANCE_PATH" >&2
  exit 1
fi
if [ -z "$BASE_COMMIT" ] || [ "$BASE_COMMIT" = "null" ]; then
  echo "Missing base_commit in $INSTANCE_PATH" >&2
  exit 1
fi

WORKDIR="${WORKDIR_ROOT:?}/${INSTANCE_ID}"
mkdir -p "$(dirname "$WORKDIR")"

if [ ! -d "$WORKDIR/.git" ]; then
  echo "Cloning https://github.com/${REPO}.git into ${WORKDIR}..."
  git clone "https://github.com/${REPO}.git" "$WORKDIR"
fi
cd "$WORKDIR"
git fetch origin "$BASE_COMMIT" 2>/dev/null || true
git checkout "$BASE_COMMIT"

mkdir -p "$(dirname "$OUTPUT_PATH")"
exec baseline \
  -orla-url "${ORLA_URL}" \
  -instance "$INSTANCE_PATH" \
  -workdir "$WORKDIR" \
  -output "$OUTPUT_PATH" \
  "$@"
