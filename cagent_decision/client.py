"""SDK-free sync/async HTTP clients for cagent workers and agent runtimes."""

from typing import Any, Dict, Optional

import httpx
from pydantic import ValidationError

from .contracts import ClassifyRequest, ClassifyResponse


class DecisionError(RuntimeError):
    """Base for explicit caller handling; no silent tool execution/fallback."""


class DecisionUnavailable(DecisionError):
    """Timeout, network failure, full queue or unavailable model."""


class DecisionRequestError(DecisionError):
    """Authentication, invalid input or another non-retryable HTTP error."""


class DecisionProtocolError(DecisionError):
    """Malformed response or an answer outside the caller's allowed labels."""


def parse_response(response: httpx.Response, request: ClassifyRequest) -> ClassifyResponse:
    """Validate both the HTTP schema and membership in the requested answer space."""
    if response.status_code in (429, 503, 502, 504):
        raise DecisionUnavailable(f"Decision API returned HTTP {response.status_code}")
    if response.status_code != 200:
        raise DecisionRequestError(f"Decision API returned HTTP {response.status_code}")
    try:
        result = ClassifyResponse.model_validate(response.json())
        if set(result.predictions) != set(request.tasks) or set(result.probabilities) != set(
            request.tasks
        ):
            raise ValueError("Task keys differ from request")
        for task, config in request.tasks.items():
            value = result.predictions[task]
            if isinstance(value, list) != config.multi_label:
                raise ValueError("Single/multi-label answer type mismatch")
            selected = value if isinstance(value, list) else [value]
            if (
                not selected
                or len(set(selected)) != len(selected)
                or not set(selected).issubset(config.labels)
            ):
                raise ValueError("Answer is outside allowed labels")
            if set(result.probabilities[task]) != set(config.labels):
                raise ValueError("Probability labels differ from request")
        return result
    except (ValueError, ValidationError, TypeError) as error:
        raise DecisionProtocolError("Invalid decision response") from error


class DecisionClient:
    """Reuse one connection pool per cagent worker process; close at shutdown."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": "Bearer " + api_key},
            transport=transport,
            trust_env=False,
        )

    def classify(self, text: str, tasks: Dict[str, Any]) -> ClassifyResponse:
        request = ClassifyRequest(text=text, tasks=tasks)
        try:
            response = self.http.post("v1/classify", json=request.model_dump(exclude_none=True))
        except httpx.RequestError as error:
            raise DecisionUnavailable("Decision API connection/timeout failure") from error
        return parse_response(response, request)

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "DecisionClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncDecisionClient:
    """Async counterpart for API handlers and asynchronous agent runtimes."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": "Bearer " + api_key},
            trust_env=False,
        )

    async def classify(self, text: str, tasks: Dict[str, Any]) -> ClassifyResponse:
        request = ClassifyRequest(text=text, tasks=tasks)
        try:
            response = await self.http.post(
                "v1/classify", json=request.model_dump(exclude_none=True)
            )
        except httpx.RequestError as error:
            raise DecisionUnavailable("Decision API connection/timeout failure") from error
        return parse_response(response, request)

    async def close(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> "AsyncDecisionClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
