.PHONY: run ui streamlit api stack test eval ingest docker-build docker-up help

API_PORT ?= 8088
WEB_PORT ?= 3500

help:
	@echo "Available commands:"
	@echo "  make run          - Start the CLI agent"
	@echo "  make api          - Start the FastAPI backend (port $(API_PORT))"
	@echo "  make ui           - Start the Next.js frontend (port $(WEB_PORT))"
	@echo "  make stack        - Start API + UI together (foreground; Ctrl-C stops both)"
	@echo "  make streamlit    - Start the legacy Streamlit UI (port 8501)"
	@echo "  make test         - Run all Python tests"
	@echo "  make eval         - Run LangSmith evaluation (skips escalation)"
	@echo "  make ingest       - Embed and push policy chunks to Qdrant"
	@echo "  make docker-build - Build Docker images (api + ui)"
	@echo "  make docker-up    - Start API + UI with docker compose"

run:
	uv run --frozen python main.py

api:
	uv run --frozen uvicorn api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

ui:
	cd frontend && NEXT_PUBLIC_API_BASE_URL=http://localhost:$(API_PORT) npm run dev -- -p $(WEB_PORT)

stack:
	@trap 'kill 0' INT TERM; \
	$(MAKE) api & \
	$(MAKE) ui & \
	wait

streamlit:
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
