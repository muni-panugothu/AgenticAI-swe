"""Application resource shutdown hooks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestResourceLifecycle(unittest.TestCase):
    def test_shutdown_closes_mongo_groq_and_embedding_clients(self):
        from app.services import resource_lifecycle

        mock_close = MagicMock()
        groq = MagicMock(close=MagicMock())
        embedding = MagicMock(close=MagicMock())
        redis = MagicMock(close=MagicMock())
        with patch(
            "app.services.memory_service.close_mongo_client",
            mock_close,
        ), patch("app.chatbot.client", groq), patch(
            "app.services.embedding_service.client", embedding
        ), patch(
            "app.services.temporary_memory_service.redis", redis
        ):
            resource_lifecycle.shutdown_application_resources()
        mock_close.assert_called_once()
        groq.close.assert_called_once()
        embedding.close.assert_called_once()
        redis.close.assert_called_once()

    def test_optional_summary_groq_client_is_closed(self):
        from app.services.memory_summary_service import _optional_llm_polish

        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="polished"))]
        )
        with patch.dict("os.environ", {"MEMORY_SUMMARY_USE_LLM": "1", "GROQ_API_KEY": "test-only"}), patch(
            "groq.Groq", return_value=client
        ):
            result = _optional_llm_polish("draft", [])

        self.assertEqual(result, "polished")
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
