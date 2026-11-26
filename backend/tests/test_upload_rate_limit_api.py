
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import AuthenticatedUser, get_current_user
from app.api.routes.documents import router
from app.services.upload_rate_limit_service import UploadRateLimitExceeded


class TestUploadRateLimitApi(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)

        async def override_current_user():
            return AuthenticatedUser(
                id="test-user-123",
                email="test@example.com",
                tenant_id="test-tenant",
                status="active",
            )

        app.dependency_overrides[get_current_user] = override_current_user
        self.client = TestClient(app)

    def _upload(self, path="/upload"):
        return self.client.post(
            f"/api/documents{path}",
            files={"file": ("test.txt", b"test document", "text/plain")},
        )

    def test_ten_uploads_allowed_and_eleventh_rejected(self):
        attempts = 0

        def fake_rate_limit(user_id):
            nonlocal attempts
            self.assertEqual(user_id, "test-user-123")
            attempts += 1

            if attempts > 10:
                raise UploadRateLimitExceeded(37)

        with (
            patch(
                "app.api.routes.documents.enforce_upload_rate_limit",
                side_effect=fake_rate_limit,
            ),
            patch(
                "app.api.routes.documents.ingest_document",
                return_value={"document_id": "test-document"},
            ),
            patch(
                "app.api.routes.documents._project_scope",
                return_value=None,
            ),
        ):
            for _ in range(10):
                response = self._upload()
                self.assertEqual(response.status_code, 200)

            response = self._upload()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "37")
        self.assertEqual(
            response.json()["detail"]["code"],
            "UPLOAD_RATE_LIMIT_EXCEEDED",
        )

    def test_both_upload_endpoints_apply_rate_limit(self):
        with (
            patch(
                "app.api.routes.documents.enforce_upload_rate_limit"
            ) as rate_limit,
            patch(
                "app.api.routes.documents.ingest_document",
                return_value={"document_id": "test-document"},
            ),
            patch(
                "app.api.routes.documents._project_scope",
                return_value=None,
            ),
        ):
            upload_response = self._upload("/upload")
            ingest_response = self._upload("/ingest")

        self.assertEqual(upload_response.status_code, 200)
        self.assertEqual(ingest_response.status_code, 200)
        self.assertEqual(rate_limit.call_count, 2)
        rate_limit.assert_any_call("test-user-123")

    def test_redis_failure_rejects_upload_with_503(self):
        with (
            patch(
                "app.api.routes.documents.enforce_upload_rate_limit",
                side_effect=RuntimeError("Redis unavailable"),
            ),
            patch(
                "app.api.routes.documents.ingest_document"
            ) as ingest,
        ):
            response = self._upload()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"],
            "UPLOAD_RATE_LIMIT_UNAVAILABLE",
        )
        ingest.assert_not_called()

    def test_rate_limit_rejection_does_not_ingest_document(self):
        with (
            patch(
                "app.api.routes.documents.enforce_upload_rate_limit",
                side_effect=UploadRateLimitExceeded(25),
            ),
            patch(
                "app.api.routes.documents.ingest_document"
            ) as ingest,
        ):
            response = self._upload()

        self.assertEqual(response.status_code, 429)
        ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
