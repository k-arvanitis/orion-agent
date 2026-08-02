.PHONY: run ui api stack demo test eval ingest seed-support docker-build docker-up help

API_PORT ?= 8088
WEB_PORT ?= 3500
DEMO_SUPPORT_DATABASE_URL ?= sqlite:///data/orion_support.db
OPENROUTER_ENV_FILE ?= ../vault-rag/.env

# Reuse only VaultRAG's OpenRouter key when that sibling repo is available.
# Other VaultRAG settings are deliberately not imported into Orion.
define load_openrouter_key
if [ -f "$(OPENROUTER_ENV_FILE)" ]; then \
	export OPENROUTER_API_KEY="$$(sed -n 's/^OPENROUTER_API_KEY=//p' "$(OPENROUTER_ENV_FILE)" | tail -n 1)"; \
	export LLM_PROVIDER=openrouter; \
fi;
endef

help:
	@echo "Available commands:"
	@echo "  make run          - Start the CLI agent"
	@echo "  make api          - Start the FastAPI backend (port $(API_PORT))"
	@echo "  make ui           - Start the Next.js frontend (port $(WEB_PORT))"
	@echo "  make stack        - Start API + UI together (foreground; Ctrl-C stops both)"
	@echo "  make demo         - Start the portfolio demo with its local SQLite database"
	@echo "  make test         - Run all Python tests"
	@echo "  make eval         - Run LangSmith evaluation"
	@echo "  make ingest       - Embed and push policy chunks to Qdrant"
	@echo "  make seed-support - Create/seed the support CRM database"
	@echo "  make docker-build - Build Docker images (api + ui)"
	@echo "  make docker-up    - Start API + UI with docker compose"

run:
	@$(load_openrouter_key) uv run --frozen python main.py

api:
	@$(load_openrouter_key) uv run --frozen uvicorn api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

ui:
	cd frontend && NEXT_PUBLIC_API_BASE_URL=http://localhost:$(API_PORT) npm run dev -- -p $(WEB_PORT)

stack: seed-support
	@trap 'kill 0' INT TERM; \
	$(MAKE) api & \
	$(MAKE) ui & \
	wait

demo:
	@SUPPORT_DATABASE_URL="$(DEMO_SUPPORT_DATABASE_URL)" $(MAKE) stack

test:
	uv run --frozen pytest tests/ -v

EVAL_EXPERIMENT ?= orion-v5

eval:
	@$(load_openrouter_key) nohup uv run --frozen python eval/run_eval.py --experiment $(EVAL_EXPERIMENT) > eval.log 2>&1 & echo "Eval PID $$! — tailing eval.log (Ctrl-C to detach, eval keeps running)"; tail -f eval.log

ingest:
	uv run --frozen python -m ingestion.chunker data/policies
	uv run --frozen python ingestion/ingest.py

seed-support:
	uv run --frozen python -m ingestion.seed_support_data

docker-build:
	docker compose build

docker-up:
	@$(load_openrouter_key) docker compose up
