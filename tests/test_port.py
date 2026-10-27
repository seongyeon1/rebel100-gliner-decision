"""Offline parity using real DeBERTa relative attention and GLiNER processor."""

from typing import Any

import pytest
import torch
from gliner2 import ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import DebertaV2Config, PreTrainedTokenizerFast


@pytest.fixture
def model() -> Any:
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "search": 5,
                "sql": 6,
                "문서": 7,
                "조회": 8,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        position_buckets=32,
        share_att_key=True,
        position_biased_input=False,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(31)
    result = BoundaryExtractor(
        ExtractorConfig(
            model_name="offline-deberta",
            architecture="boundary",
            token_pooling="first",
            boundary_head={
                "boundary_dim": 16,
                "pair_dim": 16,
                "dropout": 0.0,
                "enable_records": False,
                "enable_relations": False,
            },
        ),
        encoder_config=cfg,
        tokenizer=tokenizer,
        use_flashdeberta=False,
    )
    return result.eval()


def test_classification_wrapper_matches_public_api(model: Any) -> None:
    from port import ClassificationOnly, predict, prepare

    tasks = {
        "route": {
            "labels": ["search", "sql"],
            "label_descriptions": {"search": "문서 검색", "sql": "데이터 조회"},
        },
        "tags": {"labels": ["a", "b", "c"], "multi_label": True, "cls_threshold": 0.99},
    }
    row = {"text": "문서 조회", "tasks": tasks}

    # Executing extraction modules is a regression for this classification port.
    def reject_boundary(*args: Any) -> None:
        raise AssertionError("boundary extraction executed")

    handle = model.boundary_head.register_forward_pre_hook(reject_boundary)
    with torch.inference_mode():
        expected = model.classify_text(row["text"], tasks, include_confidence=True)
        prepared = prepare(model, row, 128)
        predictions, probabilities, _ = predict(ClassificationOnly(model), prepared, "cpu")
    handle.remove()
    for name in tasks:
        public = expected[name]
        if isinstance(public, list):
            assert predictions[name] == [item["label"] for item in public]
        else:
            assert predictions[name] == public["label"]
            assert probabilities[name][public["label"]] == pytest.approx(
                public["confidence"], abs=1e-6
            )


def test_padding_keeps_valid_marker_scores(model: Any) -> None:
    from port import ClassificationOnly, predict, prepare

    row = {"text": "문서 조회", "tasks": {"route": ["search", "sql"]}}
    with torch.inference_mode():
        a = predict(ClassificationOnly(model), prepare(model, row, 64), "cpu")
        b = predict(ClassificationOnly(model), prepare(model, row, 128), "cpu")
    assert a[0] == b[0]
    assert a[1]["route"] == pytest.approx(b[1]["route"], abs=1e-6)


def test_overflow_is_rejected_instead_of_silently_truncated(model: Any) -> None:
    from port import prepare

    with pytest.raises(ValueError, match="bucket"):
        prepare(model, {"text": "文書 " * 80, "tasks": {"route": ["search", "sql"]}}, 16)
