import os
import unittest
import base64
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from fastapi.testclient import TestClient

from app.api.deps import AuthenticatedUser, get_current_user
from app.main import app
from app.services import rag_document_service as rag


class TestRagDocumentListing(unittest.TestCase):

    def test_listing_uses_aggregation_and_returns_grouped_documents(self):
        collection = MagicMock()
        collection.aggregate.return_value = [
            {
                "document_id": "doc-1",
                "filename": "notes.txt",
                "file_type": "txt",
                "chunk_count": 12,
                "created_at": "2026-11-13T10:00:00Z",
                "owner_id": "alice",
                "permission_level": "private",
            }
        ]

        with patch.object(rag, "_collection", return_value=collection):
            result = rag.list_documents(
                user_id="alice",
                tenant_id="tenant-1",
                project_id=None,
            )

        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["documents"][0]["document_id"], "doc-1")
        self.assertEqual(result["documents"][0]["chunk_count"], 12)
        self.assertIsNone(result["next_cursor"])

        collection.aggregate.assert_called_once()
        collection.count_documents.assert_not_called()
        collection.find.assert_not_called()

    def test_aggregation_enforces_sharing_permissions(self):
        collection = MagicMock()
        collection.aggregate.return_value = []

        with patch.object(rag, "_collection", return_value=collection):
            rag.list_documents(
                user_id="alice",
                tenant_id="tenant-1",
                project_id=None,
            )

        pipeline = collection.aggregate.call_args.args[0]

        self.assertIn("$addFields", pipeline[1])
        self.assertIn("$match", pipeline[2])

        permission_match = pipeline[2]["$match"]["$or"]

        self.assertIn(
            {
                "shared_with": "alice",
                "permission_level": "shared_read",
                "tenant_id": {"$ne": None},
            },
            permission_match,
        )

    def test_aggregation_counts_chunks_and_sorts_documents(self):
        collection = MagicMock()
        collection.aggregate.return_value = []

        with patch.object(rag, "_collection", return_value=collection):
            rag.list_documents(
                user_id="alice",
                tenant_id="tenant-1",
                project_id=None,
            )

        pipeline = collection.aggregate.call_args.args[0]

        group_stage = next(
            stage["$group"]
            for stage in pipeline
            if "$group" in stage
        )

        self.assertEqual(group_stage["chunk_count"], {"$sum": 1})

        sort_stages = [
            stage["$sort"]
            for stage in pipeline
            if "$sort" in stage
        ]

        self.assertEqual(len(sort_stages), 2)
        self.assertEqual(sort_stages[0]["created_at"], -1)

    def _encode_cursor(self, payload):
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode("utf-8")
        ).decode("ascii")
        return encoded.rstrip("=")

    def _assert_invalid_cursor(self, cursor):
        collection = MagicMock()

        with patch.object(rag, "_collection", return_value=collection):
            with self.assertRaises(rag.InvalidDocumentCursor):
                rag.list_documents(
                    user_id="alice",
                    tenant_id="tenant-1",
                    project_id=None,
                    cursor=cursor,
                )

        collection.aggregate.assert_not_called()

    def test_invalid_base64_cursor_is_rejected(self):
        self._assert_invalid_cursor("%%%invalid-base64%%%")

    def test_invalid_json_cursor_is_rejected(self):
        cursor = base64.urlsafe_b64encode(
            b"{invalid-json"
        ).decode("ascii").rstrip("=")

        self._assert_invalid_cursor(cursor)

    def test_non_object_cursor_is_rejected(self):
        self._assert_invalid_cursor(self._encode_cursor(["invalid"]))

    def test_missing_cursor_fields_are_rejected(self):
        self._assert_invalid_cursor(
            self._encode_cursor({"created_at": "2026-11-16T10:00:00+00:00"})
        )

    def test_invalid_cursor_timestamp_is_rejected(self):
        self._assert_invalid_cursor(
            self._encode_cursor({
                "created_at": "not-a-date",
                "document_id": "doc-1",
                "owner_id": "alice",
            })
        )

    def test_timezone_naive_cursor_timestamp_is_rejected(self):
        self._assert_invalid_cursor(
            self._encode_cursor({
                "created_at": "2026-11-16T10:00:00",
                "document_id": "doc-1",
                "owner_id": "alice",
            })
        )

    def test_empty_cursor_identifiers_are_rejected(self):
        valid_timestamp = "2026-11-16T10:00:00+00:00"

        invalid_payloads = [
            {
                "created_at": valid_timestamp,
                "document_id": "",
                "owner_id": "alice",
            },
            {
                "created_at": valid_timestamp,
                "document_id": "doc-1",
                "owner_id": "",
            },
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self._assert_invalid_cursor(
                    self._encode_cursor(payload)
                )




class TestRagDocumentListingApi(unittest.TestCase):

    def setUp(self):
        self.user = AuthenticatedUser(
            id="alice",
            email="alice@example.com",
            tenant_id="tenant-1",
            status="active",
        )

        app.dependency_overrides[get_current_user] = lambda: self.user
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_current_user, None)

    @patch("app.api.routes.documents._project_scope", return_value=None)
    @patch(
        "app.api.routes.documents.list_document_data",
        return_value={
            "documents": [],
            "next_cursor": None,
        },
    )
    def test_default_pagination(self, mock_list, mock_project):
        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "documents": [],
                "next_cursor": None,
            },
        )

        mock_project.assert_called_once_with("alice")
        mock_list.assert_called_once_with(
            user_id="alice",
            tenant_id="tenant-1",
            project_id=None,
            page_size=20,
            cursor=None,
        )

    @patch("app.api.routes.documents._project_scope", return_value="project-1")
    @patch(
        "app.api.routes.documents.list_document_data",
        return_value={
            "documents": [{"document_id": "doc-1"}],
            "next_cursor": "next-page-token",
        },
    )
    def test_custom_pagination_and_cursor_forwarding(
        self,
        mock_list,
        mock_project,
    ):
        response = self.client.get(
            "/api/documents",
            params={
                "limit": 5,
                "cursor": "previous-page-token",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["next_cursor"],
            "next-page-token",
        )
        self.assertEqual(
            response.json()["documents"],
            [{"document_id": "doc-1"}],
        )

        mock_list.assert_called_once_with(
            user_id="alice",
            tenant_id="tenant-1",
            project_id="project-1",
            page_size=5,
            cursor="previous-page-token",
        )

    @patch("app.api.routes.documents.list_document_data")
    def test_invalid_page_size_is_rejected(self, mock_list):
        response = self.client.get(
            "/api/documents",
            params={"limit": 101},
        )

        self.assertEqual(response.status_code, 422)
        mock_list.assert_not_called()

    @patch("app.api.routes.documents._project_scope", return_value=None)
    @patch(
        "app.api.routes.documents.list_document_data",
        side_effect=rag.InvalidDocumentCursor(
            "Invalid document pagination cursor"
        ),
    )
    def test_invalid_cursor_returns_bad_request(
        self,
        mock_list,
        mock_project,
    ):
        response = self.client.get(
            "/api/documents",
            params={"cursor": "invalid-cursor"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"]["code"],
            "INVALID_DOCUMENT_CURSOR",
        )

    @patch("app.api.routes.documents._project_scope", return_value=None)
    @patch(
        "app.api.routes.documents.list_document_data",
        side_effect=ValueError("Database configuration error"),
    )
    def test_unrelated_value_error_returns_service_unavailable(
        self, mock_list, mock_project,
    ):
        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"],
            "RAG_STORE_UNAVAILABLE",
        )

    def test_pagination_generates_cursor_when_more_documents_exist(self):
        created_at = datetime(2026, 11, 13, 10, 0, tzinfo=timezone.utc)

        collection = MagicMock()
        collection.aggregate.return_value = [
            {
                "document_id": "doc-3",
                "owner_id": "alice",
                "created_at": created_at,
                "chunk_count": 3,
            },
            {
                "document_id": "doc-2",
                "owner_id": "alice",
                "created_at": created_at,
                "chunk_count": 2,
            },
            {
                "document_id": "doc-1",
                "owner_id": "alice",
                "created_at": created_at,
                "chunk_count": 1,
            },
        ]

        with patch.object(rag, "_collection", return_value=collection):
            result = rag.list_documents(
                user_id="alice",
                tenant_id="tenant-1",
                project_id=None,
                page_size=2,
            )

        self.assertEqual(
            [doc["document_id"] for doc in result["documents"]],
            ["doc-3", "doc-2"],
        )
        self.assertIsNotNone(result["next_cursor"])

        token = result["next_cursor"]
        decoded = base64.urlsafe_b64decode(
            token + "=" * (-len(token) % 4)
        )
        cursor_data = json.loads(decoded)

        self.assertEqual(cursor_data["document_id"], "doc-2")
        self.assertEqual(cursor_data["owner_id"], "alice")
        self.assertEqual(
            cursor_data["created_at"],
            created_at.isoformat(),
        )

        pipeline = collection.aggregate.call_args.args[0]
        self.assertEqual(pipeline[-1], {"$limit": 3})

    def test_invalid_service_page_size_is_rejected(self):
        with self.assertRaises(ValueError):
            rag.list_documents(
                user_id="alice",
                tenant_id="tenant-1",
                project_id=None,
                page_size=101,
            )



if __name__ == "__main__":
    unittest.main()
