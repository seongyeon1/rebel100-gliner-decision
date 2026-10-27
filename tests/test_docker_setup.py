"""Check SDK selection and env-file preservation without any real NPU."""

from pathlib import Path

import pytest


def test_sdk_selection_refuses_ambiguous_images() -> None:
    from docker_setup import choose_image

    with pytest.raises(ValueError, match="multiple"):
        choose_image({"sdk-one:1": "id1", "sdk-two:2": "id2"})


def test_sdk_selection_deduplicates_aliases() -> None:
    from docker_setup import choose_image

    assert choose_image({"sdk-one:1": "same", "sdk-one:latest": "same"}) in {
        "sdk-one:1",
        "sdk-one:latest",
    }


def test_setup_preserves_existing_key_and_user_settings(tmp_path: Path) -> None:
    from docker_setup import update_env

    path = tmp_path / ".env"
    path.write_text("DECISION_API_KEY=keep-me\nBUCKET=512\n")
    update_env(path, {"RBLN_BASE_IMAGE": "sdk:1", "DECISION_API_KEY": "new-key"})
    text = path.read_text()
    assert "DECISION_API_KEY=keep-me" in text
    assert "BUCKET=512" in text
    assert "RBLN_BASE_IMAGE=sdk:1" in text


def test_empty_sdk_is_not_accepted() -> None:
    from docker_setup import choose_image

    with pytest.raises(ValueError, match="No compatible"):
        choose_image({})
