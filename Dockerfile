# syntax=docker/dockerfile:1
# check=skip=InvalidDefaultArgInFrom
# An SDK image is mandatory; a generic default would not contain the vendor SDK.
# Pass an existing, driver-compatible Rebellions SDK image via Compose/.env.
ARG RBLN_BASE_IMAGE
FROM ${RBLN_BASE_IMAGE}
USER 0
ARG SDK_PYTHON=python3
ENV PYTHON=${SDK_PYTHON} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    SKIP_INSTALL=1 \
    HOST=0.0.0.0 \
    PORT=8009
WORKDIR /app
# The vendor image may be Debian/Ubuntu or RHEL. Git is needed for the pinned source.
RUN if command -v git >/dev/null 2>&1; then :; \
    elif command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then dnf install -y git ca-certificates && dnf clean all; \
    elif command -v microdnf >/dev/null 2>&1; then microdnf install -y git ca-certificates && microdnf clean all; \
    else echo 'Install git in the SDK base image before building' >&2; exit 1; fi
COPY requirements*.txt ./
COPY docker/install_app.py ./docker/install_app.py
RUN "$PYTHON" docker/install_app.py
# Explicit allowlist: no .env, local results, HF weights or credentials are copied.
COPY *.py *.sh *.md pyproject.toml examples.jsonl ./
COPY cagent_decision/ ./cagent_decision/
COPY docker/entrypoint.sh docker/healthcheck.sh docker/container_check.py ./docker/
RUN mkdir -p /app/results /cache/huggingface /data && chmod +x docker/*.sh
EXPOSE 8009
ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]
CMD ["serve"]
