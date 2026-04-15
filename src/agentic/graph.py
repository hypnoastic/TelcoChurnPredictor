from __future__ import annotations

import json
import os
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agentic.prompts import FOLLOW_UP_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT
from src.agentic.retriever import AgentConfigurationError, RetentionKnowledgeBase
from src.agentic.schemas import FollowUpResponse, RetentionReport
from src.agentic.state import RetentionState
from src.inference import predict_single

DEFAULT_PRIMARY_MODEL = "gemini-3-flash-preview"
DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"


class RetentionStrategist:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise AgentConfigurationError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY before using the retention strategist."
            )

        self.model_candidates = [
            os.getenv("RETENTION_MODEL", DEFAULT_PRIMARY_MODEL),
            os.getenv("RETENTION_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
        ]
        self.top_k = int(os.getenv("RETRIEVER_TOP_K", "4"))
        self.retriever = RetentionKnowledgeBase(api_key=self.api_key)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RetentionState)
        workflow.add_node("route_request", self.route_request)
        workflow.add_node("prepare_input", self.prepare_input)
        workflow.add_node("score_customer", self.score_customer)
        workflow.add_node("summarize_risk", self.summarize_risk)
        workflow.add_node("retrieve_strategies", self.retrieve_strategies)
        workflow.add_node("generate_plan", self.generate_plan)
        workflow.add_node("validate_output", self.validate_output)
        workflow.add_node("retrieve_followup_context", self.retrieve_followup_context)
        workflow.add_node("generate_follow_up", self.generate_follow_up)

        workflow.add_edge(START, "route_request")
        workflow.add_conditional_edges(
            "route_request",
            self.route_choice,
            {
                "report": "prepare_input",
                "follow_up": "retrieve_followup_context",
            },
        )
        workflow.add_edge("prepare_input", "score_customer")
        workflow.add_edge("score_customer", "summarize_risk")
        workflow.add_edge("summarize_risk", "retrieve_strategies")
        workflow.add_edge("retrieve_strategies", "generate_plan")
        workflow.add_edge("generate_plan", "validate_output")
        workflow.add_edge("validate_output", END)
        workflow.add_edge("retrieve_followup_context", "generate_follow_up")
        workflow.add_edge("generate_follow_up", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def _invoke_structured_llm(self, schema, prompt_text: str):
        last_error = None
        for model_name in self.model_candidates:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.2,
                    google_api_key=self.api_key,
                )
                structured_llm = llm.with_structured_output(schema)
                return structured_llm.invoke(prompt_text)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"All configured Gemini models failed: {last_error}")

    def route_request(self, state: RetentionState) -> RetentionState:
        request_mode = "follow_up" if state.get("follow_up_question") else "report"
        return {"request_mode": request_mode}

    def route_choice(self, state: RetentionState) -> str:
        return state.get("request_mode", "report")

    def prepare_input(self, state: RetentionState) -> RetentionState:
        if not state.get("customer_payload"):
            raise ValueError("Customer payload is required for retention report generation.")
        return {
            "follow_up_question": "",
            "model_name": state.get("model_name", "Logistic Regression"),
        }

    def score_customer(self, state: RetentionState) -> RetentionState:
        prediction = predict_single(
            state["customer_payload"],
            model_name=state.get("model_name", "Logistic Regression"),
        )
        return {"prediction": prediction}

    def summarize_risk(self, state: RetentionState) -> RetentionState:
        prediction = state["prediction"]
        profile = prediction["customer_profile"]
        probability = float(prediction["probability"])
        risk_band = "high" if probability >= 0.75 else "medium" if probability >= 0.5 else "low"

        business_factors = []
        if profile["Contract"] == "Month-to-month":
            business_factors.append("month-to-month contract")
        if profile["tenure"] <= 12:
            business_factors.append("short tenure")
        if profile["PaymentMethod"] == "Electronic check":
            business_factors.append("electronic check payment friction")
        if profile["TechSupport"] == "No":
            business_factors.append("no tech support add-on")
        if profile["OnlineSecurity"] == "No":
            business_factors.append("no online security add-on")
        if profile["MonthlyCharges"] >= 70:
            business_factors.append("higher monthly charges")

        summary = (
            f"{risk_band.title()} churn risk at {prediction['probability_pct']} using "
            f"{state.get('model_name', 'Logistic Regression')}. "
            f"Top model drivers: {', '.join(prediction['drivers'])}. "
            f"Observed business factors: {', '.join(business_factors) if business_factors else 'limited obvious risk factors'}."
        )

        query_parts = [summary]
        if state.get("user_query"):
            query_parts.append(state["user_query"])
        return {
            "risk_summary": summary,
            "retrieval_query": " ".join(query_parts),
        }

    def retrieve_strategies(self, state: RetentionState) -> RetentionState:
        retrieved = self.retriever.retrieve(state["retrieval_query"], top_k=self.top_k)
        return {"retrieved_context": retrieved}

    def generate_plan(self, state: RetentionState) -> RetentionState:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REPORT_SYSTEM_PROMPT),
                (
                    "human",
                    "Customer payload:\n{customer_payload}\n\n"
                    "Prediction summary:\n{prediction}\n\n"
                    "Risk summary:\n{risk_summary}\n\n"
                    "User retention query:\n{user_query}\n\n"
                    "Retrieved evidence:\n{retrieved_context}",
                ),
            ]
        ).format_messages(
            customer_payload=json.dumps(state["customer_payload"], indent=2),
            prediction=json.dumps(state["prediction"], indent=2),
            risk_summary=state["risk_summary"],
            user_query=state.get("user_query", "No extra retention query supplied."),
            retrieved_context=json.dumps(state["retrieved_context"], indent=2),
        )
        report = self._invoke_structured_llm(RetentionReport, prompt)
        return {
            "retention_report": report.model_dump(),
            "follow_up_history": [],
        }

    def validate_output(self, state: RetentionState) -> RetentionState:
        report = state.get("retention_report", {})
        if not report.get("recommended_actions"):
            raise ValueError("Structured retention report did not include recommended actions.")
        if not report.get("retrieved_evidence"):
            raise ValueError("Structured retention report did not include retrieved evidence.")
        return {}

    def retrieve_followup_context(self, state: RetentionState) -> RetentionState:
        if not state.get("retention_report"):
            raise ValueError("Generate a retention report before asking follow-up questions.")
        follow_up_question = state.get("follow_up_question", "")
        query = " ".join(
            [
                follow_up_question,
                state.get("risk_summary", ""),
                " ".join(state.get("retention_report", {}).get("priority_order", [])),
            ]
        ).strip()
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        return {"retrieved_context": retrieved}

    def generate_follow_up(self, state: RetentionState) -> RetentionState:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FOLLOW_UP_SYSTEM_PROMPT),
                (
                    "human",
                    "Customer payload:\n{customer_payload}\n\n"
                    "Existing retention report:\n{retention_report}\n\n"
                    "Follow-up question:\n{follow_up_question}\n\n"
                    "Retrieved evidence:\n{retrieved_context}\n\n"
                    "Conversation history:\n{follow_up_history}",
                ),
            ]
        ).format_messages(
            customer_payload=json.dumps(state["customer_payload"], indent=2),
            retention_report=json.dumps(state["retention_report"], indent=2),
            follow_up_question=state["follow_up_question"],
            retrieved_context=json.dumps(state["retrieved_context"], indent=2),
            follow_up_history=json.dumps(state.get("follow_up_history", []), indent=2),
        )
        response = self._invoke_structured_llm(FollowUpResponse, prompt)
        history = list(state.get("follow_up_history", []))
        history.append(
            {
                "question": state["follow_up_question"],
                "answer": response.answer,
            }
        )
        return {
            "follow_up_response": response.model_dump(),
            "follow_up_history": history,
            "follow_up_question": "",
        }

    def generate_report(
        self,
        customer_payload: dict,
        user_query: str = "",
        thread_id: str | None = None,
        model_name: str = "Logistic Regression",
    ) -> tuple[str, dict, dict, list[dict[str, str]]]:
        active_thread_id = thread_id or str(uuid4())
        result = self.graph.invoke(
            {
                "customer_payload": customer_payload,
                "user_query": user_query,
                "model_name": model_name,
                "follow_up_question": "",
            },
            config={"configurable": {"thread_id": active_thread_id}},
        )
        return (
            active_thread_id,
            result["retention_report"],
            result["prediction"],
            result.get("retrieved_context", []),
        )

    def answer_follow_up(self, thread_id: str, question: str) -> tuple[dict, list[dict[str, str]]]:
        result = self.graph.invoke(
            {"follow_up_question": question},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["follow_up_response"], result.get("retrieved_context", [])
