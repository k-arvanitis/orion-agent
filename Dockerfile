FROM python:3.11-slim

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies as a cached layer (before copying source)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source, then install the project itself (skipped above so the
# dependency layer stays cached across source-only changes).
COPY . .
RUN uv sync --frozen --no-dev

# Non-root runtime user. UID 1000 matches the default first user on most
# single-user Linux dev boxes, so the ./data bind mount in docker-compose.yml
# stays writable without extra chown steps; a different host UID would need
# one (not handled here — this is a demo image, not a multi-host deployment).
RUN useradd -m -u 1000 orion && chown -R orion:orion /app
USER orion

EXPOSE 8088

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD uv run --frozen --no-sync python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/api/health', timeout=2)" || exit 1

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8088"]
