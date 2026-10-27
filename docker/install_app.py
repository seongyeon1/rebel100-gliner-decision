"""Install application requirements without replacing the vendor's SDK/PyTorch."""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

SDK_PACKAGES = ("torch", "torch-rbln", "rebel-compiler")


def main() -> None:
    """Fail at build time if the base image has no compatible Python SDK."""
    if sys.version_info < (3, 11):
        raise RuntimeError("SDK base image must provide Python >=3.11")
    try:
        before = {name: version(name) for name in SDK_PACKAGES}
    except Exception as error:
        raise RuntimeError(
            "Base image lacks the SDK. Set RBLN_BASE_IMAGE and, if needed, "
            "SDK_PYTHON to the vendor environment interpreter."
        ) from error
    constraints = Path("/app/sdk-constraints.txt")
    constraints.write_text("".join(f"{name}=={value}\n" for name, value in before.items()))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-c",
            str(constraints),
            "-r",
            "requirements-serving.txt",
        ],
        check=True,
    )
    after = {name: version(name) for name in SDK_PACKAGES}
    if before != after:
        raise RuntimeError("Vendor SDK versions changed during installation")
    print("Preserved SDK versions:", after)


if __name__ == "__main__":
    main()
