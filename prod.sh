#!/usr/bin/env bash
set -e

case "$1" in
  --build|-b)
    docker compose build
    ;;
  --down|-d)
    docker compose down
    exit 0
    ;;
  --logs|-l)
    docker compose logs -f
    exit 0
    ;;
esac

echo "🚀 Starting RAG Platform in PRODUCTION mode"
docker compose up -d
echo "✅ Running at http://localhost:3002"
