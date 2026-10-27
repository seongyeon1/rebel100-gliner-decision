"""GLiNER classification-only tensor boundary with CPU schema handling."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "fastino/GLiNER2.5-multi-Decide"
MODEL_REVISION = "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"


@dataclass
class Prepared:
    """Fixed-shape CPU inputs plus original label routing metadata."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    groups: List[Tuple[str, List[int], Dict[str, Any]]]
    tokens: int


def load_model(model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> Any:
    """Resolve a pinned complete snapshot, including its tokenizer."""
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download

    location = (
        model_id if Path(model_id).is_dir() else snapshot_download(model_id, revision=revision)
    )
    model = (
        AutoExtractor.from_pretrained(location, attention_backend="standard").cpu().float().eval()
    )
    if model.config.architecture != "boundary" or model.processor.token_pooling != "first":
        raise ValueError("This port requires boundary architecture and token_pooling=first")
    return model


class ClassificationOnly(nn.Module):
    """Score every token, moving label gather out of the compiled graph.

    Encoder relative-attention gathers remain unchanged. Dropout must be off.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.classifier = model.classifier
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )[0]
        return self.classifier(hidden).squeeze(-1)


def prepare(model: Any, row: Dict[str, Any], bucket: int) -> Prepared:
    """Use original processor without text truncation; fail on bucket overflow."""
    schema = model._classification_schema(deepcopy(row["tasks"]))
    schemas, _ = model._build_schema_dicts_and_metadata([schema])
    batch = model.processor.collate_fn_inference(
        [(row["text"], schemas[0])],
        max_len=None,
        architecture="boundary",
        error_policy="raise",
        build_targets=False,
    )
    length = batch.input_ids.shape[1]
    if length > bucket:
        raise ValueError(
            f"Input needs {length} tokens including schema; bucket={bucket}. "
            "Use a larger bucket or explicitly chunk upstream; no silent truncation."
        )
    if batch.query_marker_mask.any():
        raise ValueError("Extraction queries are not supported by the classification-only port")
    groups = []
    for i, kind in enumerate(batch.task_types[0]):
        if kind != "classifications":
            raise ValueError(f"Unexpected task type {kind}")
        tokens = batch.schema_tokens_list[0][i]
        config = model._resolve_classification_config(
            tokens[2], batch.original_schemas[0]["classifications"]
        )
        positions = list(batch.schema_special_indices[0][i][1:])
        # schema_special_indices are indices into the subword encoder sequence.
        if config is None or len(positions) != len(config["labels"]):
            raise ValueError("Classification marker/label mismatch")
        groups.append((config["task"], positions, deepcopy(config)))
    pad = model.processor.tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Tokenizer has no pad token")
    return Prepared(
        F.pad(batch.input_ids, (0, bucket - length), value=pad),
        F.pad(batch.attention_mask, (0, bucket - length), value=0),
        groups,
        length,
    )


def decode(
    logits: torch.Tensor, prepared: Prepared, temperature: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Match original GLiNER activation, threshold, fallback and top_k rules."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    predictions: Dict[str, Any] = {}
    probabilities: Dict[str, Any] = {}
    for task, positions, cfg in prepared.groups:
        scores = logits[0, positions].float() / temperature
        multi = cfg.get("multi_label", False)
        activation = cfg.get("class_act", "auto")
        probs = (
            scores.sigmoid()
            if activation == "sigmoid" or (activation == "auto" and multi)
            else scores.softmax(-1)
        )
        labels = cfg["labels"]
        probabilities[task] = dict(zip(labels, probs.tolist(), strict=False))
        if multi:
            selected = [
                j
                for j, value in enumerate(probs.tolist())
                if value >= cfg.get("cls_threshold", 0.5)
            ]
            if not selected:
                selected = [int(probs.argmax())]
            if cfg.get("top_k") is not None:
                selected = sorted(selected, key=lambda j: float(probs[j]), reverse=True)[
                    : cfg["top_k"]
                ]
            predictions[task] = [labels[j] for j in selected]
        else:
            predictions[task] = labels[int(probs.argmax())]
    return predictions, probabilities


def predict(
    module: Any, prepared: Prepared, device: str, temperature: float = 1.0
) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
    """Include input/output transfers, synchronizing with CPU logit materialization."""
    ids = prepared.input_ids.to(device)
    mask = prepared.attention_mask.to(device)
    logits = module(ids, mask).detach().cpu().float()
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Non-finite candidate logits; precision/compilation requires investigation"
        )
    prediction, probabilities = decode(logits, prepared, temperature)
    return prediction, probabilities, logits
