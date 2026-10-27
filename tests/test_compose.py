"""Validate rendered Compose behavior with the real Docker Compose parser."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_compose_cdi_and_client_isolation() -> None:
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is not installed")
    env = dict(
        os.environ,
        RBLN_BASE_IMAGE="validation-sdk:local",
        DECISION_API_KEY="validation-only",
        BIND_ADDRESS="127.0.0.1",
        PORT="8009",
        RBLN_CDI_DEVICE="rebellions.ai/npu=0",
    )
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "--profile",
            "tools",
            "--profile",
            "client",
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    services = json.loads(result.stdout)["services"]
    for name in ("api", "benchmark", "demo"):
        assert services[name]["devices"][0]["source"] == "rebellions.ai/npu=0"
        assert not services[name].get("privileged", False)
    assert not services["client"].get("devices")
    assert services["client"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["api"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["benchmark"]["environment"]["SKIP_INSTALL"] == "1"


def test_build_context_excludes_credentials_and_outputs() -> None:
    text = Path("Dockerfile").read_text()
    assert "COPY . " not in text
    ignored = Path(".dockerignore").read_text().splitlines()
    for path in (".env", ".git", "results", "models", "data", "*.log"):
        assert path in ignored
