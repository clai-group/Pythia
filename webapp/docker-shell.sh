#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="pythia-webapp"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the dev image
docker build -t "$IMAGE_NAME" -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"

# Run dev server with source mounted for live reload
docker run --rm -it \
  -p 3000:3000 \
  -v "$ROOT_DIR/webapp:/app" \
  -v pythia-webapp-node-modules:/app/node_modules \
  -e CHOKIDAR_USEPOLLING=true \
  "$IMAGE_NAME" "$@"
