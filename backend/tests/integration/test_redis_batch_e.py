"""Batch E — disposable Redis namespace integration (gated)."""

from __future__ import annotations

import json
import time
import unittest
import uuid
from unittest.mock import patch

from tests.integration.batch_e_safety import new_batch_e_run_id, redis_gate

_GATE = redis_gate()


@unittest.skipUnless(_GATE.enabled, _GATE.reason)
class TestBatchERedisLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.services import temporary_memory_service as tms

        cls.redis = tms.redis
        cls.run_id = new_batch_e_run_id()
        cls.ns = _GATE.namespace.rstrip(":") + f":{cls.run_id}:"
        try:
            probe = cls.ns + "health"
            cls.redis.set(probe, "1", ex=30)
            if cls.redis.get(probe) is None:
                raise unittest.SkipTest(
                    "NOT RUN: Redis set/get probe failed for Batch E namespace"
                )
        except unittest.SkipTest:
            raise
        except Exception as exc:
            raise unittest.SkipTest(
                f"NOT RUN: Redis unavailable ({type(exc).__name__})"
            ) from exc

    def tearDown(self):
        self._cleanup_namespace()

    @classmethod
    def tearDownClass(cls):
        try:
            pattern = cls.ns + "*"
            cursor = 0
            while True:
                cursor, keys = cls.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    cls.redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass

    def _cleanup_namespace(self):
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

    def test_BE_RD01_health_ttl_and_namespace(self):
        key = self.ns + f"ttl:{uuid.uuid4().hex}"
        self.redis.set(key, "alive", ex=2)
        self.assertEqual(self.redis.get(key), "alive")
        time.sleep(2.2)
        self.assertIsNone(self.redis.get(key))

    def test_BE_RD02_rate_limit_per_user_isolation(self):
        from app.core.config import get_settings
        from app.services.chat_rate_limit_service import (
            ChatRateLimitExceeded,
            chat_rate_limit_key,
            enforce_chat_rate_limit,
        )

        settings = get_settings()
        user_a = f"be_redis_a_{uuid.uuid4().hex[:8]}"
        user_b = f"be_redis_b_{uuid.uuid4().hex[:8]}"
        with patch.object(settings, "chat_rate_limit_count", 2), patch.object(
            settings, "chat_rate_limit_window_seconds", 60
        ):
            enforce_chat_rate_limit(user_a)
            enforce_chat_rate_limit(user_a)
            with self.assertRaises(ChatRateLimitExceeded):
                enforce_chat_rate_limit(user_a)
            enforce_chat_rate_limit(user_b)
        for uid in (user_a, user_b):
            try:
                self.redis.delete(chat_rate_limit_key(uid))
            except Exception:
                pass

    def test_BE_RD03_task_context_roundtrip_and_clear(self):
        from app.services import temporary_memory_service as tms

        user_id = f"be_ctx_{uuid.uuid4().hex[:8]}"
        saved = tms.add_task_context(user_id, "user", f"batche {self.run_id}")
        self.assertTrue(saved.get("success"))
        loaded = tms.get_task_context(user_id)
        self.assertIn(self.run_id, json.dumps(loaded))
        self.assertTrue(tms.clear_task_context(user_id))
        self.assertEqual(tms.get_task_context(user_id), [])

    def test_BE_RD04_fail_closed_when_redis_eval_errors(self):
        from app.services import chat_rate_limit_service as crl

        user = f"be_fail_{uuid.uuid4().hex[:8]}"

        with patch.object(
            crl.redis,
            "eval",
            side_effect=RuntimeError("redis down"),
        ):
            # Fail-closed: Redis errors must not silently allow unlimited chat.
            with self.assertRaises(Exception):
                crl.enforce_chat_rate_limit(user)


if __name__ == "__main__":
    unittest.main()
