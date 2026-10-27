"""Discover an already available SDK image and write a local Compose env file."""

import json
import os
import re
import secrets
import subprocess
from pathlib import Path
from typing import Dict, Optional, Set

ROOT = Path(__file__).resolve().parent
PROBE = (
    "import sys,importlib.metadata as m; assert sys.version_info >= (3,11); "
    'print({k:m.version(k) for k in ("torch","torch-rbln","rebel-compiler")})'
)


def read_env(path: Path) -> Dict[str, str]:
    """Read simple dotenv values as data, never as shell commands."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def update_env(path: Path, values: Dict[str, str], replace: Optional[Set[str]] = None) -> None:
    """Keep existing keys/settings; do not rotate an existing API key."""
    replace = replace or set()
    existing = read_env(path)
    lines = path.read_text().splitlines() if path.exists() else []
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise ValueError("Multiline env values are not supported")
        if existing.get(key) and key not in replace:
            continue
        lines = [line for line in lines if line.split("=", 1)[0].strip() != key]
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def choose_image(compatible: Dict[str, str]) -> str:
    """Only choose automatically when all references identify the same image."""
    if not compatible:
        raise ValueError(
            "No compatible local SDK image found. Obtain a driver-compatible "
            "SDK image from Rebellions or set RBLN_BASE_IMAGE explicitly."
        )
    if len(set(compatible.values())) != 1:
        raise ValueError(
            "Found multiple SDK images; set RBLN_BASE_IMAGE to one of: " + ", ".join(compatible)
        )
    return next(iter(compatible))


def discover_images(python: str) -> Dict[str, str]:
    """Probe running-image references and locally tagged Rebellions/SDK images."""
    running = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Image}}"], text=True
    ).splitlines()
    tagged = subprocess.check_output(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"], text=True
    ).splitlines()
    candidates = list(
        dict.fromkeys(running + [x for x in tagged if re.search(r"rbln|rebellions", x, re.I)])
    )
    compatible = {}
    for image in candidates:
        if "<none>" in image:
            continue
        check = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--entrypoint",
                python,
                image,
                "-c",
                PROBE,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check.returncode == 0:
            image_id = subprocess.check_output(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
            ).strip()
            compatible[image] = image_id
    return compatible


def resolve_image(existing: Dict[str, str], python: str) -> str:
    """Reuse an SDK image or bootstrap one only when discovery finds none."""
    image = os.environ.get("RBLN_BASE_IMAGE") or existing.get("RBLN_BASE_IMAGE")
    if image:
        return image
    found = discover_images(python)
    if found:
        return choose_image(found)
    if (os.getenv("AUTO_BUILD_SDK") or existing.get("AUTO_BUILD_SDK", "1")) != "1":
        return choose_image({})
    from sdk_bootstrap import build_sdk

    image = build_sdk(ROOT, existing)
    # The bootstrap image exposes its venv on PATH as python3.
    os.environ["SDK_PYTHON"] = "python3"
    return image


def main() -> None:
    """Keep host Docker/CDI configuration untouched; only prepare project files."""
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "compose", "version"], check=True)
    path = ROOT / ".env"
    existing = read_env(path)
    python = os.environ.get("SDK_PYTHON") or existing.get("SDK_PYTHON") or "python3"
    image = resolve_image(existing, python)
    python = os.environ.get("SDK_PYTHON") or python
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*", image):
        raise ValueError("Invalid RBLN_BASE_IMAGE reference")
    if not path.exists():
        path.write_text((ROOT / ".env.example").read_text())
    update_env(
        path,
        {
            "RBLN_BASE_IMAGE": image,
            "SDK_PYTHON": python,
            "DECISION_API_KEY": os.environ.get("DECISION_API_KEY") or secrets.token_urlsafe(32),
        },
        replace={key for key in ("RBLN_BASE_IMAGE", "SDK_PYTHON") if os.environ.get(key)},
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    subprocess.run(["docker", "compose", "config", "--quiet"], cwd=ROOT, check=True)
    print(
        json.dumps(
            {
                "sdk_base_image": image,
                "env_file": str(path),
                "note": "API key retained/generated in .env; key not printed. Host CDI must already be configured.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
