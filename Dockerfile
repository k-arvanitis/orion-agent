FROM python:3.11-slim

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies as a cached layer (before copying source)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY . .

EXPOSE 8501

CMD ["uv", "run", "--frozen", "streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
