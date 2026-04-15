# Customer Churn Project Status Report

## Executive Summary

The project now delivers both required milestones in one hosted system:

- **Milestone 1**: churn prediction with Logistic Regression and Decision Tree
- **Milestone 2**: LangGraph-based agentic retention strategy with local RAG and structured outputs

The final app is not just a classifier. It predicts churn, explains the main drivers, retrieves retention playbooks, and proposes grounded intervention actions.

## Current ML Model Summary

The deployed application uses two production models.

| Model | Accuracy | Precision | Recall | F1 Score | Role in App |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Logistic Regression | 79.2% | 59.4% | 68.5% | 63.6% | Default scoring model and agent input |
| Decision Tree | 74.7% | 51.6% | 76.4% | 61.6% | Dashboard comparison model |

### Recommendation

Use **Logistic Regression** as the default business model because it offers the best overall balance for the hosted workflow and produces clearer driver explanations for the agent.

Use **Decision Tree** as a comparison model when recall-first business review is needed.

## Milestone 1 Completion Status

Implemented:

- preprocessing with scaling and categorical encoding
- EDA and visual diagnostics
- Logistic Regression and Decision Tree training
- evaluation metrics and confusion matrices
- single-customer prediction form
- batch CSV scoring with downloadable output
- churn-driver explanation summary
- hosted Gradio deployment

## Milestone 2 Completion Status

Implemented:

- LangGraph workflow with explicit state
- FAISS-backed local retrieval
- telecom retention playbooks in the knowledge base
- Gemini-based structured report generation
- follow-up Q&A on the same customer case
- Hugging Face secret-based API key configuration

Structured output fields:

- business context
- risk summary
- key drivers
- retrieved evidence
- recommended actions
- priority order
- next-touch plan
- confidence notes

## Deployment Notes

- GitHub `main` contains the full project history and tracked model artifacts.
- The Hugging Face Space is deployed through a separate Space branch to avoid binary push restrictions.
- If deployment artifacts are absent in the Space branch, the app can bootstrap training at startup.

## Final Assessment

The project now satisfies the assignment progression from classical ML prediction to agentic retention strategy. The strongest outcome is that both milestones are connected through one shared inference layer rather than two disconnected demos.
