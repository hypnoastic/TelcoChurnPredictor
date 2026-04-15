from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class RetentionState(TypedDict, total=False):
    request_mode: str
    customer_payload: dict[str, Any]
    model_name: str
    user_query: str
    follow_up_question: str
    prediction: dict[str, Any]
    risk_summary: str
    retrieval_query: str
    retrieved_context: list[dict[str, str]]
    retention_report: dict[str, Any]
    follow_up_response: dict[str, Any]
    follow_up_history: list[dict[str, str]]
    error: str
