"""Install the minimal SDK combination documented for RBLN SDK 0.12.0.

No driver or NPU is accessed during the build. Metadata checks intentionally
avoid importing torch, which can autoload the device extension before CDI exists.
"""

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import List

SDK_PINS = {"rebel-compiler": "0.12.0", "torch": "2.13.0+cpu", "torch-rbln": "0.12.0"}


def installation_commands(config_present: bool) -> List[List[str]]:
    """Do not overwrite credential-bearing index URLs from a pip secret."""
    compiler_index = [] if config_present else ["--extra-index-url", "https://pypi.rbln.ai/simple"]
    return [
        [*compiler_index, "rebel-compiler==0.12.0"],
        ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.13.0+cpu"],
        ["torch-rbln==0.12.0"],
    ]


def main() -> None:
    """Install the compiler first, followed by its paired CPU torch/extension."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The bootstrap image requires Ubuntu 24.04 Python 3.12")
    config = os.environ.get("PIP_CONFIG_FILE", "")
    commands = installation_commands(bool(config) and Path(config).is_file())
    for args in commands:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-input", *args],
            check=True,
        )
    found = {name: version(name) for name in SDK_PINS}
    if found != SDK_PINS:
        raise RuntimeError(f"Unexpected SDK version pairing: {found}")
    print("SDK metadata validated:", found)


if __name__ == "__main__":
    main()
