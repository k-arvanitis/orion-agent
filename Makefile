.PHONY: run ui api stack test eval ingest docker-build docker-up help

API_PORT ?= 8088
WEB_PORT ?= 3500

help:
	@echo "Available commands:"
	@echo "  make run          - Start the CLI agent"
	@echo "  make api          - Start the FastAPI backend (port $(API_PORT))"
	@echo "  make ui           - Start the Next.js frontend (port $(WEB_PORT))"
	@echo "  make stack        - Start API + UI together (foreground; Ctrl-C stops both)"
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

test:
	uv run --frozen pytest tests/ -v

EVAL_EXPERIMENT ?= orion-v5

eval:
	nohup uv run --frozen python eval/run_eval.py --skip-escalation --experiment $(EVAL_EXPERIMENT) > eval.log 2>&1 & echo "Eval PID $$! — tailing eval.log (Ctrl-C to detach, eval keeps running)"; tail -f eval.log

ingest:
	uv run --frozen python -m ingestion.chunker data/policies
	uv run --frozen python ingestion/ingest.py

docker-build:
	docker compose build

docker-up:
	docker compose up
