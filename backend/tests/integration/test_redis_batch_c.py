"""Batch C — Redis rate-limit / task-context integration (gated)."""

from __future__ import annotations

import json
import time
import unittest
import uuid
from unittest.mock import patch

from tests.integration.safety import redis_gate

_GATE = redis_gate()


@unittest.skipUnless(_GATE.enabled, _GATE.reason)
class TestBatchCRedisIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.services import temporary_memory_service as tms

        cls.redis = tms.redis
        cls.ns = _GATE.namespace.rstrip(":") + ":"
        try:
            probe = cls.ns + "health"
            cls.redis.set(probe, "1", ex=30)
            val = cls.redis.get(probe)
            if val is None:
                raise unittest.SkipTest(
                    "NOT RUN: Redis set/get probe failed for configured namespace"
                )
        except unittest.SkipTest:
            raise
        except Exception as exc:
            raise unittest.SkipTest(
                f"NOT RUN: Redis unavailable ({type(exc).__name__})"
            ) from exc

    def tearDown(self):
        pattern = self.ns + "*"
        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    self.redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass

    def test_BC_RD01_namespace_isolation_and_ttl(self):
        key = self.ns + f"ttl:{uuid.uuid4().hex}"
        self.redis.set(key, "alive", ex=2)
        self.assertEqual(self.redis.get(key), "alive")
        time.sleep(2.2)
        self.assertIsNone(self.redis.get(key))

    def test_BC_RD02_chat_rate_limit_per_user(self):
        from app.core.config import get_settings
        from app.services.chat_rate_limit_service import (
            ChatRateLimitExceeded,
            chat_rate_limit_key,
            enforce_chat_rate_limit,
        )

        settings = get_settings()
        user_a = f"bc_redis_a_{uuid.uuid4().hex[:8]}"
        user_b = f"bc_redis_b_{uuid.uuid4().hex[:8]}"
        with patch.object(settings, "chat_rate_limit_count", 3), patch.object(
            settings, "chat_rate_limit_window_seconds", 60
        ):
            for _ in range(3):
                enforce_chat_rate_limit(user_a)
            with self.assertRaises(ChatRateLimitExceeded):
                enforce_chat_rate_limit(user_a)
            enforce_chat_rate_limit(user_b)
        for uid in (user_a, user_b):
            try:
                self.redis.delete(chat_rate_limit_key(uid))
            except Exception:
                pass

    def test_BC_RD03_task_context_serialization(self):
        from app.services import temporary_memory_service as tms

        user_id = f"bc_ctx_{uuid.uuid4().hex[:8]}"
        saved = tms.add_task_context(user_id, "user", "batchc hello")
        self.assertTrue(saved.get("success"))
        loaded = tms.get_task_context(user_id)
        self.assertIn("batchc hello", json.dumps(loaded))
        self.assertTrue(tms.clear_task_context(user_id))
        after = tms.get_task_context(user_id)
        self.assertEqual(after, [])


if __name__ == "__main__":
    unittest.main()
