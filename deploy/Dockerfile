# LUMOS web interface: a single image (backend and interface), built from the root of the repository:
#
#   docker build -f deploy/Dockerfile -t lumos .
#   docker run --rm -p 127.0.0.1:8000:8000 lumos          # http://localhost:8000, local profile
#
# The local profile serves one user on their own machine: the container listens on all its interfaces
# (--listen-all) and its port must be published on 127.0.0.1 only. For a public instance, see
# deploy/public/ and the hosting guide (docs/web/hosting.md).

# --- Interface (Vite build) -------------------------------------------------------------------------------
FROM node:22-bookworm-slim AS interface
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN mkdir -p /src/src/lumos/web && npm run build

# --- Python package ---------------------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS lumos
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/lumos
COPY . .
COPY --from=interface /src/src/lumos/web/static src/lumos/web/static
# hatch-vcs reads the version from git; without history, a fixed one
ARG LUMOS_VERSION=0.0.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${LUMOS_VERSION} pip install ".[web,mcp,catalogs]"

RUN useradd --create-home --uid 10001 lumos && mkdir -p /data && chown lumos /data
USER lumos
WORKDIR /home/lumos
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
ENTRYPOINT ["lumos", "web"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--listen-all", "--archive-root", "/data/archives"]
