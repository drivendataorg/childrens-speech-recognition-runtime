## BASE IMAGE
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Install system dependencies
COPY apt.txt /tmp/apt.txt
RUN apt-get update --fix-missing \
    # && apt-get install -y apt-utils software-properties-common curl 2> /dev/null \
    # && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    # && apt-get install -y \
    #     libopenblas0 \
    #     libgomp1 \
    && xargs -a /tmp/apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/apt.txt

# Create runtime user
ENV RUNTIME_USER=runtimeuser
ENV RUNTIME_UID=1000
ENV RUNTIME_GID=1000

RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_GID} ${RUNTIME_USER} \
    && useradd --create-home --gid ${RUNTIME_GID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER}

# Set up code execution working directory
RUN mkdir /code_execution && \
    chown -R ${RUNTIME_USER}:${RUNTIME_USER} /code_execution
WORKDIR /code_execution

# Switch to runtime user for Python package installation
USER ${RUNTIME_USER}

# Copy files for environment
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} \
    .python-version \
    pyproject.toml \
    uv.lock \
    ./

# uv environment variables
ENV UV_NO_DEV=1 \
    UV_FROZEN=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies
# Use cache mount to speed up rebuilds
ENV UV_CACHE_DIR=/home/runtimeuser/.cache/uv
ENV UV_PYTHON_CACHE_DIR=/home/runtimeuser/.cache/uv/python
RUN --mount=type=cache,id=uv-cache,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync

# Copy entrypoint and tests
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} \
    entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

## RUNTIME
FROM base AS runtime
ENV UV_NO_SYNC=1
CMD ["bash", "/entrypoint.sh"]

## TEST-BASE
FROM base AS test-base
ENV UV_PYTHON_CACHE_DIR=/home/runtimeuser/.cache/uv/python
RUN --mount=type=cache,id=uv-cache,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra test
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} tests/ ./tests/

## TEST-RUNTIME
FROM test-base AS test-runtime
ENV UV_NO_SYNC=1
CMD ["uv", "run", "pytest", "tests"]
