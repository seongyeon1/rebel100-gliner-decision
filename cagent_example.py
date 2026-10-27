"""Run document classification and allowed-tool routing through the HTTP clients."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from cagent_decision import AsyncDecisionClient, DecisionClient, DecisionError


def document_type(client: DecisionClient, text: str, existing_type: str = "") -> str:
    """Example doc-worker integration: preserve a human-assigned document type."""
    if existing_type:
        return existing_type
    result = client.classify(text, {"document_type": ["계약서", "회의록", "보고서", "기타"]})
    return result.predictions["document_type"]


async def route_request(base_url: str, key: str) -> Dict[str, Any]:
    """The caller supplies only authorized tools; this example executes none."""
    async with AsyncDecisionClient(base_url, key) as client:
        result = await client.classify(
            "사내 계약서의 해지 조건을 찾아줘",
            {
                "route": {
                    "labels": ["document_search", "sql_query", "clarify"],
                    "label_descriptions": {
                        "document_search": "사내 문서 내용 검색",
                        "sql_query": "정형 데이터베이스 조회 및 집계",
                        "clarify": "필수 정보 추가 확인",
                    },
                }
            },
        )
        return result.model_dump()


def main() -> int:
    """Print real responses; service errors are failures, not successful examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("DECISION_API_URL", "http://127.0.0.1:8009")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = os.environ.get("DECISION_API_KEY", "")
    if not key:
        parser.error("Set DECISION_API_KEY")
    try:
        with DecisionClient(args.base_url, key) as client:
            doc = client.classify(
                "회의 참석자와 안건 및 결정 사항을 정리한 문서입니다.",
                {"document_type": ["계약서", "회의록", "보고서", "기타"]},
            )
        route = asyncio.run(route_request(args.base_url, key))
    except DecisionError as error:
        print(f"Decision API example failed: {error}")
        return 1
    result = {
        "status": "completed",
        "document_classification": doc.model_dump(),
        "agent_routing": route,
        "note": "Selections only; no tools executed. "
        "Probabilities are not calibrated confidence. This is not an accuracy benchmark.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
