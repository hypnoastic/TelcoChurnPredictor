import os

import pytest

from src.agentic.graph import AgentConfigurationError, RetentionStrategist


def test_retention_strategist_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(AgentConfigurationError):
        RetentionStrategist()
