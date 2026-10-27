"""Build an SDK base image locally when no vendor SDK image is available."""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

DEFAULT_IMAGE = "rbln-gliner-sdk:0.12.0-py312"


def pip_secret(root: Path) -> Optional[Path]:
    """Use explicit/standard pip config only through an external BuildKit secret."""
    explicit = os.getenv("SDK_PIP_CONFIG") or os.getenv("PIP_CONFIG_FILE")
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [Path.home() / ".config/pip/pip.conf", Path.home() / ".pip/pip.conf"]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_file():
            if explicit:
                raise ValueError("SDK_PIP_CONFIG does not name a readable pip config file")
            continue
        if candidate.is_relative_to(root.resolve()):
            raise ValueError("SDK_PIP_CONFIG must be outside the Docker build context")
        if any(char in str(candidate) for char in (",", "\n", "\r")):
            raise ValueError("Unsupported secret file path")
        return candidate
    return None


def host_preflight(cdi_device: str) -> None:
    """Verify real CDI/driver access before downloading/building SDK packages."""
    print("Checking host NPU access through CDI before SDK installation...", flush=True)
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--device",
                cdi_device,
                "--entrypoint",
                "/bin/sh",
                "ubuntu:24.04",
                "-ec",
                "command -v rbln-smi >/dev/null; rbln-smi",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Host driver/CDI preflight failed. Install the vendor driver and "
            "RBLN Container Toolkit or fix the CDI handle first. No SDK build "
            "was started. This script does not replace drivers or restart Docker."
        ) from error


def build_sdk(root: Path, config: Optional[Dict[str, str]] = None) -> str:
    """Build with pinned public/portal SDK wheels; never include auth in image layers."""
    config = config or {}
    image = os.getenv("SDK_BOOTSTRAP_IMAGE") or config.get("SDK_BOOTSTRAP_IMAGE") or DEFAULT_IMAGE
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*", image):
        raise ValueError("Invalid SDK_BOOTSTRAP_IMAGE")
    secret = pip_secret(root)
    cdi = os.getenv("RBLN_CDI_DEVICE") or config.get("RBLN_CDI_DEVICE") or "rebellions.ai/npu=0"
    host_preflight(cdi)
    command = [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "--progress",
        "plain",
        "-f",
        str(root / "Dockerfile.sdk"),
        "-t",
        image,
    ]
    if secret:
        command += ["--secret", f"id=sdk_pip_config,src={secret}"]
        print(
            "Using external pip configuration as a BuildKit secret; credentials are not printed.",
            flush=True,
        )
    else:
        print(
            "No pip credential config found. Trying package access without credentials; "
            "Portal-protected downloads may fail.",
            flush=True,
        )
    command.append(str(root))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "SDK image build failed. If the RBLN index reports 401/403 or a "
            "missing protected package, obtain Portal access and rerun with "
            "SDK_PIP_CONFIG=/absolute/path/outside/project/pip.conf. "
            "The script cannot bypass vendor access requirements."
        ) from error
    print(f"SDK image ready: {image}", flush=True)
    return image


def main() -> None:
    """Explicitly rebuild the SDK, then configure the normal Compose workflow."""
    from docker_setup import ROOT, read_env
    from docker_setup import main as setup

    image = build_sdk(ROOT, read_env(ROOT / ".env"))
    os.environ["RBLN_BASE_IMAGE"] = image
    os.environ["SDK_PYTHON"] = "python3"
    setup()


if __name__ == "__main__":
    main()
