FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps: required for packages that compile on arm64 (e.g. gemmi)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install package (includes console script: `boltz`)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime defaults: mount cache at /cache and point BOLTZ_CACHE there
ENV BOLTZ_CACHE=/cache

ENTRYPOINT ["boltz"]
CMD ["--help"]
