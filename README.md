# RAG Platform (Agentic, Local-First)

A production-oriented Retrieval-Augmented Generation (RAG) platform built with FastAPI, ChromaDB, Ollama, and a static frontend.

It supports:
- Multi-tenant knowledge bases with role-based access.
- Local LLM inference through Ollama.
- Hybrid retrieval with an agentic flow.
- Contextual retrieval with late chunking summaries.
- Adaptive route selection (`retrieve`, `sparse`, `graph`) and optional web fallback.
- Semantic response cache and persistent long-term memory profiles.
- OCR and layout-aware document ingestion.
- Async ingestion queue with background workers (uploads return quickly).
- Admin panel for users, groups, permissions, personalities, and KB settings.
- Audit logging for key actions.

## Tech Stack

- Backend: FastAPI, SQLAlchemy, LangChain, LangGraph
- Vector store: ChromaDB
- LLM runtime: Ollama
- Embeddings: SentenceTransformers (local)
- Frontend: HTML/CSS/Vanilla JS (served by Nginx)
- Orchestration: Docker Compose

## Architecture

1. Documents are uploaded to a knowledge base.
2. Backend classifies document complexity (`text`, `structured`, `visual`).
3. Parser pipeline is selected:
- `text`: standard text extraction + chunking.
- `structured`: Docling parser for tables/sections.
- `visual`: VLM parser for scanned/image-heavy content.
4. Upload is queued (`processing`) and background workers handle indexing.
5. Summaries/chunks are embedded and stored in Chroma.
6. User asks a question in chat.
7. Agentic retrieval + generation runs with KB-specific config/personality.
8. Answer, sources, and reasoning trace are returned and stored in session history.

## Features

- Authentication with JWT.
- First registered user becomes admin.
- Group-based KB permissions (`read`, `manage`).
- KB-level model and retrieval controls.
- Upload limits for multi-file ingestion.
- Session-based chat history.
- Audit events with actor details.

## Repository Layout

- `backend/app/main.py`: API app startup and router registration.
- `backend/app/routers/`: auth, chat, documents, admin, groups, permissions.
- `backend/app/services/`: RAG, agentic graph, parsers, audit logger.
- `backend/app/models/`: SQLAlchemy models.
- `backend/app/schemas/`: request/response schemas.
- `frontend/`: static UI and admin panel.
- `docker-compose.yml`: production-like stack.
- `docker-compose.dev.yml`: live-sync development overrides.
- `deploy/windows/`: Windows helper scripts.

## Prerequisites

- Docker Desktop (Linux containers enabled).
- NVIDIA GPU optional, but recommended for inference speed.
- NVIDIA Container Toolkit support in Docker (for GPU passthrough).
- On Windows, valid SSL cert files if running HTTPS via frontend container.

## Quick Start (Docker)

1. Create `.env` from `.env.example`.
2. Adjust values as needed (at least `SECRET_KEY`, admin defaults, SSL path).
3. Start stack:

```bash
docker compose up -d --build
```

4. Open:
- App: `https://localhost` or configured host.
- API docs: `http://localhost:8000/docs` (if backend exposed in your setup).

## Development Mode

Run with live file sync/watch:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --watch
```

Helper scripts are also included:
- `dev.sh`
- `dev.ps1`

## Environment Variables

Core:
- `SECRET_KEY`
- `DATABASE_URL`
- `CHROMA_HOST`, `CHROMA_PORT`
- `OLLAMA_HOST`, `OLLAMA_PORT`
- `DEFAULT_LLM_MODEL`

Upload and retrieval:
- `MAX_UPLOAD_FILES`
- `MAX_UPLOAD_FILE_MB`
- `MAX_UPLOAD_BATCH_MB`
- `INGEST_WORKERS`
- `INGEST_QUEUE_MAX`
- `HYBRID_LEXICAL_MAX_DOCS`
- `ENABLE_CROSS_ENCODER_RERANK`
- `CROSS_ENCODER_MODEL`
- `VECTOR_DB_PROVIDER` (`chroma` or `qdrant`)
- `QDRANT_HOST`, `QDRANT_PORT`
- `ENABLE_CONTEXTUAL_RETRIEVAL`
- `ENABLE_GRAPH_RAG`, `GRAPH_MEMORY_DIR`

Latency tuning:
- `FAST_MODE`
- `QUERY_CACHE_TTL_SEC`
- `QUERY_CACHE_MAX`
- `ENABLE_SEMANTIC_CACHE`
- `SEMANTIC_CACHE_SIMILARITY`
- `SEMANTIC_CACHE_TTL_SEC`
- `SEMANTIC_CACHE_MAX`
- `ENABLE_LONG_TERM_MEMORY`, `MEMORY_STORE_DIR`
- `ENABLE_WEB_FALLBACK`, `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_TIMEOUT_SEC`
- `VLM_MAX_PAGES`
- `VLM_DPI`
- `VLM_CONCURRENCY`
- `VLM_TIMEOUT_SEC`
- `VISUAL_MAX_PDF_PAGES`
- `VISUAL_MAX_PDF_MB`

Audit logging:
- `AUDIT_LOG_PATH`
- `AUDIT_LOG_LEVEL`

## GPU Configuration

`docker-compose.yml` includes NVIDIA GPU reservations for `ollama` and `gpus: all` for `backend`.

Validation commands:

```bash
docker inspect rag_ollama --format '{{json .HostConfig.DeviceRequests}}'
docker exec rag_ollama ollama ps
```

When a model is active, `ollama ps` should show `PROCESSOR` as GPU.

## Audit Logging

Audit logger is configured in:
- `backend/app/services/audit.py`

Startup wiring:
- `backend/app/main.py`

Logged events include:
- User registration/login outcomes.
- KB create/update/delete.
- Personality create/update/delete.
- Group create/update/delete and permission/member changes.
- Document upload start/success/failure and deletes.
- Chat session create/delete and message outcomes.

To view logs:

```bash
docker logs -f rag_backend
docker exec -it rag_backend sh -lc "tail -f /app/data/audit.log"
```

## API Overview

Auth:
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

Chat:
- `GET /chat/knowledge-bases`
- `GET /chat/kb-permission/{kb_id}`
- `POST /chat/message`
- `GET /chat/sessions`
- `GET /chat/sessions/{session_id}/messages`
- `DELETE /chat/sessions/{session_id}`

Documents:
- `POST /documents/upload/{kb_id}`
- `POST /documents/upload-multiple/{kb_id}`
- `GET /documents/{kb_id}`
- `DELETE /documents/{doc_id}`

Admin:
- `GET /admin/stats`
- `GET /admin/ollama/models`
- Personalities CRUD under `/admin/personalities`
- Knowledge base CRUD under `/admin/knowledge-bases`
- User management under `/admin/users`
- Group management under `/admin/groups`

## Troubleshooting

Slow indexing or answers:
- Use `FAST_MODE=true`.
- Keep `ENABLE_CROSS_ENCODER_RERANK=false`.
- Use lighter embedding model (`all-MiniLM-L6-v2`).
- Reduce VLM cost via `VLM_MAX_PAGES`, `VLM_DPI`, and `VLM_CONCURRENCY`.

GPU not used:
- Confirm Docker GPU support on host.
- Check `ollama ps` while a request is running.
- Use models that fit your VRAM.

Document upload fails:
- Check supported extension and upload size limits.
- Check backend logs for parser-specific exceptions.

Upload endpoint is still slow:
- Ensure backend is restarted with latest async queue worker code.
- Confirm upload returns `processing` quickly and indexing finishes later as `ready`.
- Check queue/worker activity in backend logs (`document.upload_queued` then `document.upload_completed`).
- Increase `INGEST_WORKERS` carefully if host CPU/GPU has headroom.

Frontend shows stale code:
- Ensure you are running in watch mode for dev, or rebuild frontend image in prod mode.

## Continuous Evaluation

Run the CI-oriented evaluator with a JSONL dataset:

```bash
python -m app.eval.continuous_eval --dataset ../eval/sample_eval.jsonl
```

Use threshold flags (`--min-retrieval`, `--min-grounding`, `--min-relevance`) to enforce regression gates.

## Security Notes

- Change `SECRET_KEY` in production.
- Do not commit real credentials in `.env`.
- Restrict CORS and exposed ports for deployment.
- Terminate TLS correctly and manage certificates securely.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
