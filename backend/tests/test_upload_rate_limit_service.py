
import unittest
from unittest.mock import patch

from app.services import upload_rate_limit_service as rate_limit_service
from app.services.upload_rate_limit_service import (
    UploadRateLimitExceeded,
    enforce_upload_rate_limit,
)


class TestUploadRateLimitService(unittest.TestCase):
    def test_ten_attempts_allowed_and_eleventh_rejected(self):
        counter = 0

        def fake_eval(script, keys, args):
            nonlocal counter
            counter += 1
            return [counter, 60]

        with patch.object(
            rate_limit_service.redis,
            "eval",
            side_effect=fake_eval,
        ):
            for _ in range(10):
                enforce_upload_rate_limit("test-user-123")

            with self.assertRaises(UploadRateLimitExceeded) as context:
                enforce_upload_rate_limit("test-user-123")

        self.assertEqual(counter, 11)
        self.assertEqual(context.exception.retry_after_seconds, 60)

    def test_redis_uses_user_specific_key_and_configured_window(self):
        with patch.object(
            rate_limit_service.redis,
            "eval",
            return_value=[1, 60],
        ) as mock_eval:
            enforce_upload_rate_limit("user-abc")

        mock_eval.assert_called_once()

        args = mock_eval.call_args
        self.assertIn("INCR", args.args[0])
        self.assertIn("EXPIRE", args.args[0])
        self.assertEqual(args.kwargs["keys"], ["rag:upload:rate:user-abc"])
        self.assertEqual(
            args.kwargs["args"],
            [str(rate_limit_service.settings.upload_rate_limit_window_seconds)],
        )

    def test_empty_user_id_is_rejected(self):
        with patch.object(rate_limit_service.redis, "eval") as mock_eval:
            with self.assertRaises(ValueError):
                enforce_upload_rate_limit(" ")

        mock_eval.assert_not_called()

    def test_unexpected_redis_response_is_rejected(self):
        with patch.object(
            rate_limit_service.redis,
            "eval",
            return_value=[1],
        ):
            with self.assertRaises(RuntimeError):
                enforce_upload_rate_limit("user-abc")

    def test_redis_errors_propagate(self):
        with patch.object(
            rate_limit_service.redis,
            "eval",
            side_effect=ConnectionError("Redis unavailable"),
        ):
            with self.assertRaises(ConnectionError):
                enforce_upload_rate_limit("user-abc")


if __name__ == "__main__":
    unittest.main()
