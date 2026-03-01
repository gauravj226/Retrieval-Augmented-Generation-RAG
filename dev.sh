#!/usr/bin/env bash
set -e

# ── RAG Platform — Dev launcher ───────────────────────────────────────────────
# Merges docker-compose.yml (base) with docker-compose.dev.yml (dev overrides)
# and starts everything in watch mode.
#
# First run:  ./dev.sh --build
# Normal run: ./dev.sh

COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"

case "$1" in
  --build|-b)
    echo "🔨 Building dev images..."
    docker compose $COMPOSE_FILES build
    ;;
  --down|-d)
    echo "🛑 Stopping dev stack..."
    docker compose $COMPOSE_FILES down
    exit 0
    ;;
  --logs|-l)
    docker compose $COMPOSE_FILES logs -f backend frontend
    exit 0
    ;;
  --clean|-c)
    echo "🗑  Removing containers and volumes..."
    docker compose $COMPOSE_FILES down -v
    exit 0
    ;;
esac

echo ""
echo "🚀 Starting RAG Platform in DEVELOPMENT mode"
echo "   Backend  → http://localhost:8000"
echo "   Frontend → http://localhost:3002"
echo "   Docs     → http://localhost:8000/docs"
echo ""
echo "   File watch is ACTIVE:"
echo "   • backend/app/**    → synced → uvicorn auto-reloads"
echo "   • frontend/**       → synced → browser-refresh to see"
echo "   • requirements.txt  → triggers full rebuild"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

docker compose $COMPOSE_FILES up --watch
