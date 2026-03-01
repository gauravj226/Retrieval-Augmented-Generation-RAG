#!/bin/sh
ollama serve &
SERVE_PID=$!
echo "[ollama-init] Waiting for Ollama to start..."
sleep 8

echo "[ollama-init] Pulling llama3.2:3b (System 2 Agentic RAG model)..."
ollama pull llama3.2:3b

echo "[ollama-init] Pulling qwen2.5vl:3b (visual document pipeline)..."
ollama pull qwen2.5vl:3b

echo "[ollama-init] Pulling nomic-embed-text..."
ollama pull nomic-embed-text

echo "[ollama-init] All models ready."
wait $SERVE_PID
