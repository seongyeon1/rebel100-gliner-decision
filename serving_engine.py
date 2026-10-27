"""Single-thread model lifetime and classification-only NPU execution."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import torch

from cagent_decision.contracts import ClassifyRequest
from port import MODEL_ID, MODEL_REVISION, ClassificationOnly, load_model, predict, prepare


@dataclass(frozen=True)
class Settings:
    """One serving process owns one device and one fixed token bucket."""

    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    backend: Literal["cpu", "rbln-compile", "rbln-eager"] = "rbln-compile"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    bucket: int = 256
    device: int = 0
    cpu_threads: int = 4
    max_pending: int = 8
    api_key: str = ""
    cache_dir: str = "results/serve-cache"

    @classmethod
    def from_env(cls) -> "Settings":
        """Use existing benchmark env names and dedicated HTTP settings."""
        return cls(
            model=os.getenv("MODEL", MODEL_ID),
            revision=os.getenv("REVISION", MODEL_REVISION),
            backend=os.getenv("BACKEND", "rbln-compile"),
            dtype=os.getenv("DTYPE", "float16"),
            bucket=int(os.getenv("BUCKET", "256")),
            device=int(os.getenv("NPU_DEVICE", "0")),
            cpu_threads=int(os.getenv("CPU_THREADS", "4")),
            max_pending=int(os.getenv("MAX_PENDING", "8")),
            api_key=os.getenv("DECISION_API_KEY", ""),
            cache_dir=os.getenv("SERVE_CACHE_DIR", "results/serve-cache"),
        )

    def validate(self) -> None:
        """Fail startup on invalid settings rather than silently using CPU."""
        if self.backend not in ("cpu", "rbln-compile", "rbln-eager"):
            raise ValueError("Unsupported BACKEND")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported DTYPE")
        if min(self.bucket, self.cpu_threads, self.max_pending) < 1 or self.device < 0:
            raise ValueError("Invalid bucket/threads/queue/device setting")
        if not self.api_key:
            raise ValueError("DECISION_API_KEY is required")


class Engine:
    """Create and warm the model in the same dedicated thread used for inference."""

    @torch.inference_mode()
    def __init__(self, settings: Settings) -> None:
        settings.validate()
        torch.set_num_threads(settings.cpu_threads)
        self.settings = settings
        self.model = load_model(settings.model, settings.revision)
        self.temperature = float(self.model.boundary_settings.classification_temperature)
        self.device = "cpu" if settings.backend == "cpu" else f"rbln:{settings.device}"
        if settings.backend != "cpu":
            import rebel  # noqa: F401
            import torch_rbln  # noqa: F401

            if not torch.rbln.is_available():
                raise RuntimeError("RBLN NPU unavailable")
        self.module: Any = (
            ClassificationOnly(self.model)
            .to(device=self.device, dtype=getattr(torch, settings.dtype))
            .eval()
        )
        if settings.backend == "rbln-compile":
            cache = Path(settings.cache_dir).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            self.module = torch.compile(
                self.module,
                backend="rbln",
                fullgraph=True,
                dynamic=False,
                options={"mode": "strict", "cache_dir": str(cache)},
            )
        self.classify(ClassifyRequest(text="문서 분류", tasks={"warmup": ["a", "b"]}))

    @torch.inference_mode()
    def classify(self, request: ClassifyRequest) -> Dict[str, Any]:
        """Use CPU schema handling and the same tensor boundary as the benchmark."""
        row = request.model_dump(exclude_none=True)
        prepared = prepare(self.model, row, self.settings.bucket)
        predictions, probabilities, _ = predict(
            self.module, prepared, self.device, self.temperature
        )
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "tokens": prepared.tokens,
            "model_revision": self.settings.revision,
            "backend": self.settings.backend,
        }
