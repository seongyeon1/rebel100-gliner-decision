"""HTTP contracts shared by the serving process and SDK-free cagent clients."""

import math
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskConfig(BaseModel):
    """A bounded classification task; labels stay under caller control."""

    model_config = ConfigDict(extra="forbid")
    labels: List[str] = Field(min_length=1, max_length=64)
    label_descriptions: Dict[str, str] = Field(default_factory=dict)
    multi_label: bool = False
    cls_threshold: float = Field(default=0.5, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=64)
    class_act: Literal["auto", "sigmoid", "softmax"] = "auto"
    prompt: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_labels(self) -> "TaskConfig":
        """Reject duplicate, blank or unexpectedly large labels/descriptions."""
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Duplicate labels")
        if any(not label.strip() or len(label) > 240 for label in self.labels):
            raise ValueError("Label must contain 1..240 characters")
        if not set(self.label_descriptions).issubset(self.labels):
            raise ValueError("Description key is not a label")
        if any(len(value) > 2000 for value in self.label_descriptions.values()):
            raise ValueError("Label description exceeds 2000 characters")
        return self


class ClassifyRequest(BaseModel):
    """Only classification tasks are accepted; no arbitrary extraction schemas."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=128000)
    tasks: Dict[str, TaskConfig] = Field(min_length=1, max_length=16)

    @field_validator("tasks", mode="before")
    @classmethod
    def expand_short_form(cls, value: Any) -> Any:
        """Accept the original API's task: [label, ...] shorthand."""
        if isinstance(value, dict):
            return {
                key: {"labels": config} if isinstance(config, list) else config
                for key, config in value.items()
            }
        return value

    @model_validator(mode="after")
    def validate_content(self) -> "ClassifyRequest":
        """Reject blank requests and invalid task names."""
        if not self.text.strip() or any(not k.strip() or len(k) > 120 for k in self.tasks):
            raise ValueError("Text and task names must be nonblank")
        return self


class ClassifyResponse(BaseModel):
    """Typed predictions and raw model probabilities, not calibrated confidence."""

    model_config = ConfigDict(extra="forbid")
    request_id: str
    predictions: Dict[str, Union[str, List[str]]]
    probabilities: Dict[str, Dict[str, float]]
    model_revision: str
    backend: str
    tokens: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def finite_values(self) -> "ClassifyResponse":
        """Reject NaN/Inf or invalid probability ranges in HTTP responses."""
        if not math.isfinite(self.elapsed_ms):
            raise ValueError("Non-finite elapsed time")
        if any(
            not math.isfinite(p) or not 0 <= p <= 1
            for values in self.probabilities.values()
            for p in values.values()
        ):
            raise ValueError("Invalid probabilities")
        return self
