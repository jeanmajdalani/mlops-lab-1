
# =====================================
# STAGE 1: BUILDER
# =====================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /usr/local/bin/

# Copy CPU-only dependencies
COPY docker-cpu/pyproject.toml ./pyproject.toml
COPY docker-cpu/uv.lock ./uv.lock

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project


# =====================================
# STAGE 2: RUNTIME
# =====================================

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv

# Copy only application source
COPY src/food11/serve.py ./src/food11/serve.py

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
