"""Exercise the real classification engine through HTTP and the cagent client."""

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_port import model  # noqa: F401


def test_http_client_matches_direct_classification(model: Any, tmp_path: Path) -> None:  # noqa: F811
    from cagent_decision import DecisionClient
    from server import Settings, create_app

    model.save_pretrained(str(tmp_path / "model"))
    settings = Settings(
        model=str(tmp_path / "model"),
        backend="cpu",
        dtype="float32",
        bucket=128,
        api_key="test-key",
    )
    with TestClient(create_app(settings)) as http:
        for _ in range(100):
            if http.get("/readyz").status_code == 200:
                break
            time.sleep(0.02)
        assert http.get("/readyz").status_code == 200
        assert (
            http.post("/v1/classify", json={"text": "x", "tasks": {"r": ["a"]}}).status_code == 401
        )
        with DecisionClient(
            "http://testserver", "test-key", transport=httpx_transport(http)
        ) as client:
            result = client.classify("文書 조회", {"route": ["search", "sql"]})
        expected = model.classify_text("文書 조회", {"route": ["search", "sql"]})
        assert result.predictions == expected
        assert set(result.probabilities["route"]) == {"search", "sql"}
        invalid = http.post(
            "/v1/classify",
            headers={"Authorization": "Bearer test-key"},
            json={"text": "x", "tasks": {"r": ["a", "a"]}},
        )
        assert invalid.status_code == 422


def httpx_transport(http: Any) -> Any:
    """Bridge an httpx client to a live ASGI test server without mocking inference."""
    import httpx

    return httpx.MockTransport(
        lambda request: http.request(
            request.method, str(request.url), content=request.content, headers=dict(request.headers)
        )
    )


def test_client_rejects_unknown_predicted_tool() -> None:
    import httpx

    from cagent_decision import DecisionClient, DecisionProtocolError

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "request_id": "1",
                "predictions": {"route": "delete_everything"},
                "probabilities": {"route": {"search": 1.0}},
                "model_revision": "x",
                "backend": "cpu",
                "tokens": 1,
                "elapsed_ms": 1.0,
            },
        )
    )
    with DecisionClient("http://test", "key", transport=transport) as client:
        with pytest.raises(DecisionProtocolError):
            client.classify("search", {"route": ["search"]})


def test_real_http_server_and_cagent_demo(model: Any, tmp_path: Path) -> None:  # noqa: F811
    import json
    import os
    import subprocess
    import sys

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    output = tmp_path / "http-demo"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        BACKEND="cpu",
        DTYPE="float32",
        SKIP_INSTALL="1",
        MODEL=str(checkpoint),
        BUCKET="128",
        CPU_THREADS="1",
    )
    result = subprocess.run(
        ["bash", "run_serving_demo.sh", "--output", str(output), "--startup-timeout", "30"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "demo-report.json").read_text())
    responses = json.loads((output / "responses.json").read_text())
    assert report["status"] == "completed"
    assert responses["document_classification"]["backend"] == "cpu"
    assert responses["agent_routing"]["predictions"]["route"] in {
        "document_search",
        "sql_query",
        "clarify",
    }


def test_model_load_failure_keeps_readiness_false() -> None:
    from server import Settings, create_app

    def fail(settings: Any) -> Any:
        raise RuntimeError("failure details must stay in server logs")

    with TestClient(create_app(Settings(api_key="key"), fail)) as http:
        for _ in range(100):
            ready = http.get("/readyz")
            if "startup_failed" in ready.text:
                break
            time.sleep(0.01)
        assert ready.status_code == 503
        assert "failure details" not in ready.text
        response = http.post(
            "/v1/classify",
            headers={"Authorization": "Bearer key"},
            json={"text": "x", "tasks": {"route": ["search"]}},
        )
        assert response.status_code == 503


def test_overflow_returns_422(model: Any, tmp_path: Path) -> None:  # noqa: F811
    from server import Settings, create_app

    model.save_pretrained(str(tmp_path / "model"))
    with TestClient(
        create_app(
            Settings(
                model=str(tmp_path / "model"),
                backend="cpu",
                dtype="float32",
                bucket=32,
                api_key="key",
            )
        )
    ) as http:
        for _ in range(100):
            if http.get("/readyz").status_code == 200:
                break
            time.sleep(0.02)
        response = http.post(
            "/v1/classify",
            headers={"Authorization": "Bearer key"},
            json={"text": "long " * 100, "tasks": {"r": ["a", "b"]}},
        )
        assert response.status_code == 422


def test_queue_saturation_returns_429() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from server import Settings, create_app

    entered, release = threading.Event(), threading.Event()

    class BlockingEngine:
        """A controlled slow backend for observable admission-control behavior."""

        def __init__(self, settings: Any) -> None:
            pass

        def classify(self, request: Any) -> Any:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test deadline exceeded")
            return {
                "predictions": {"r": "a"},
                "probabilities": {"r": {"a": 1.0}},
                "tokens": 1,
                "model_revision": "test",
                "backend": "cpu",
            }

    with TestClient(create_app(Settings(api_key="key", max_pending=1), BlockingEngine)) as http:
        for _ in range(100):
            if http.get("/readyz").status_code == 200:
                break
            time.sleep(0.01)

        def request() -> Any:
            return http.post(
                "/v1/classify",
                headers={"Authorization": "Bearer key"},
                json={"text": "x", "tasks": {"r": ["a"]}},
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(request)
            assert entered.wait(timeout=2)
            second = pool.submit(request)
            # Wait for the bounded queue to fill, using the readiness metrics surface.
            for _ in range(100):
                if http.get("/readyz").json().get("queued_requests") == 1:
                    break
                time.sleep(0.01)
            try:
                assert request().status_code == 429
            finally:
                release.set()
            assert first.result(timeout=2).status_code == 200
            assert second.result(timeout=2).status_code == 200


def test_client_service_failure_is_explicit() -> None:
    import httpx

    from cagent_decision import DecisionClient, DecisionUnavailable

    with DecisionClient(
        "http://test", "key", transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        with pytest.raises(DecisionUnavailable):
            client.classify("x", {"r": ["a"]})
