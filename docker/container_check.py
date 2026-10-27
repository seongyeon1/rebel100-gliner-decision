"""Check device visibility inside the container; never silently fall back to CPU."""

import os
from importlib.metadata import version


def main() -> None:
    """Import the SDK only at runtime, after CDI has injected host libraries."""
    import torch

    backend = os.environ.get("BACKEND", "rbln-compile")
    if backend not in ("cpu", "rbln-compile", "rbln-eager"):
        raise RuntimeError("Unsupported BACKEND")
    print("PyTorch:", torch.__version__)
    if backend == "cpu":
        print("Explicit CPU mode; this does not verify NPU access.")
        return
    import rebel  # noqa: F401
    import torch_rbln  # noqa: F401

    for name in ("torch-rbln", "rebel-compiler"):
        print(name, version(name))
    if not torch.rbln.is_available():
        raise RuntimeError(
            "NPU unavailable inside container. Check host driver, RBLN "
            "Container Toolkit CDI spec and RBLN_CDI_DEVICE."
        )
    device = int(os.getenv("NPU_DEVICE", "0"))
    print("NPU:", torch.rbln.get_device_name(device))


if __name__ == "__main__":
    main()
