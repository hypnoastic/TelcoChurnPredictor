from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: str = Field(..., description="Knowledge-base file name used for support.")
    insight: str = Field(..., description="Retention insight taken from retrieval.")


class RecommendedAction(BaseModel):
    priority: int = Field(..., description="1-based priority order.")
    action: str = Field(..., description="Retention action to take.")
    rationale: str = Field(..., description="Why this action fits the customer profile.")
    owner: str = Field(..., description="Suggested team owner.")
    timeline: str = Field(..., description="Execution timing.")


class RetentionReport(BaseModel):
    business_context: str
    risk_summary: str
    key_drivers: list[str]
    retrieved_evidence: list[EvidenceItem]
    recommended_actions: list[RecommendedAction]
    priority_order: list[str]
    next_touch_plan: list[str]
    confidence_notes: str


class FollowUpResponse(BaseModel):
    answer: str
    supporting_points: list[str]
    cited_sources: list[str]
