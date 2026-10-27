"""SDK bootstrap must preserve credentials and perform host checks before builds."""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_secret_must_stay_outside_build_context(tmp_path: Path) -> None:
    from sdk_bootstrap import pip_secret

    root = tmp_path / "project"
    root.mkdir()
    secret = root / "pip.conf"
    secret.write_text("secret-data")
    with patch.dict("os.environ", {"SDK_PIP_CONFIG": str(secret)}, clear=True):
        with pytest.raises(ValueError, match="outside"):
            pip_secret(root)


def test_sdk_pins_match_documented_release() -> None:
    from docker.bootstrap_install import SDK_PINS

    assert SDK_PINS == {"rebel-compiler": "0.12.0", "torch": "2.13.0+cpu", "torch-rbln": "0.12.0"}


def test_fallback_builds_only_when_no_sdk_image() -> None:
    from docker_setup import resolve_image

    with (
        patch("docker_setup.discover_images", return_value={}),
        patch("sdk_bootstrap.build_sdk", return_value="rbln-gliner-sdk:test") as build,
    ):
        assert resolve_image({}, "python3") == "rbln-gliner-sdk:test"
        build.assert_called_once()


def test_existing_image_skips_bootstrap() -> None:
    from docker_setup import resolve_image

    with patch("sdk_bootstrap.build_sdk") as build:
        assert resolve_image({"RBLN_BASE_IMAGE": "provided-sdk:1"}, "python3") == "provided-sdk:1"
        build.assert_not_called()


def test_authenticated_index_config_is_not_overridden() -> None:
    from docker.bootstrap_install import installation_commands

    assert installation_commands(True)[0] == ["rebel-compiler==0.12.0"]
    assert "--extra-index-url" in installation_commands(False)[0]


def test_failed_host_check_never_starts_sdk_build(tmp_path: Path) -> None:
    from sdk_bootstrap import build_sdk

    with (
        patch("sdk_bootstrap.pip_secret", return_value=None),
        patch("sdk_bootstrap.host_preflight", side_effect=RuntimeError("CDI missing")),
        patch("sdk_bootstrap.subprocess.run") as run,
    ):
        with pytest.raises(RuntimeError, match="CDI missing"):
            build_sdk(tmp_path)
        run.assert_not_called()
