"""Authenticated classification API with bounded admission and serialized inference."""

import asyncio
import logging
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException

from cagent_decision.contracts import ClassifyRequest, ClassifyResponse
from serving_engine import Engine, Settings

log = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None, engine_factory: Callable[[Settings], Any] = Engine
) -> FastAPI:
    """Build the app; startup readiness remains false until model warmup completes."""
    config = settings or Settings.from_env()
    config.validate()
    state: Dict[str, Any] = {"ready": False, "startup_failed": False}
    queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_pending)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision-npu")

    async def worker() -> None:
        loop = asyncio.get_running_loop()
        try:
            engine = await loop.run_in_executor(executor, engine_factory, config)
        except Exception:
            log.exception("decision_model_startup_failed")
            state["startup_failed"] = True
            return
        state["ready"] = True
        while True:
            request, future = await queue.get()
            try:
                result = await loop.run_in_executor(executor, engine.classify, request)
                if not future.done():
                    future.set_result(result)
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(worker())
        try:
            yield
        finally:
            state["ready"] = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    app = FastAPI(title="GLiNER Decision API", lifespan=lifespan)

    def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not secrets.compare_digest(authorization or "", "Bearer " + config.api_key):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.get("/healthz")
    async def health() -> Dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready() -> Dict[str, Any]:
        if not state["ready"]:
            raise HTTPException(
                status_code=503,
                detail={"status": "startup_failed" if state["startup_failed"] else "warming"},
            )
        return {
            "status": "ready",
            "backend": config.backend,
            "bucket": config.bucket,
            "queued_requests": queue.qsize(),
        }

    @app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(authorize)])
    async def classify(request: ClassifyRequest) -> ClassifyResponse:
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="Model not ready")
        started = time.perf_counter()
        future = asyncio.get_running_loop().create_future()
        try:
            queue.put_nowait((request, future))
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Inference queue full") from None
        # A disconnected/timed-out caller must not cancel an in-flight NPU operation.
        try:
            result = await asyncio.shield(future)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            log.exception("decision_inference_failed")
            raise HTTPException(status_code=503, detail="Inference failed") from error
        return ClassifyResponse(
            request_id=str(uuid.uuid4()),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            **result,
        )

    return app


if __name__ == "__main__":
    import uvicorn

    # One process per assigned device; never spawn several model-owning workers.
    uvicorn.run(
        "server:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8009")),
        workers=1,
    )
