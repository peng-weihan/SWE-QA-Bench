FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY SWE-QA-Bench ./SWE-QA-Bench
COPY clone_repos.sh ./clone_repos.sh

# Install dependencies with uv (add --extra baseline if needed)
RUN uv sync --frozen
RUN sh clone_repos.sh

# Set Python path
ENV PYTHONPATH=/app
