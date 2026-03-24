.PHONY: run ui test eval ingest docker-build docker-up docker-setup help

help:
	@echo "Available commands:"
	@echo "  make run          - Start the CLI agent"
	@echo "  make ui           - Start the Streamlit UI"
	@echo "  make test         - Run all tests"
	@echo "  make eval         - Run LangSmith evaluation (skips escalation)"
	@echo "  make ingest       - Embed and push policy chunks to Qdrant"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start UI + Ollama with docker compose"
	@echo "  make docker-setup - Pull nomic-embed-text into the Ollama container"

run:
	uv run --frozen python main.py

ui:
	uv run --frozen streamlit run ui/app.py

test:
	uv run --frozen pytest tests/ -v

eval:
	uv run --frozen python eval/run_eval.py --skip-escalation --experiment orion-v1

ingest:
	uv run --frozen python -m ingestion.chunker data/policies
	uv run --frozen python ingestion/ingest.py

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-setup:
	docker compose exec ollama ollama pull nomic-embed-text
