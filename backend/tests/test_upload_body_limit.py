
import unittest
import json
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.middleware import UploadBodyLimitMiddleware


class TestUploadBodyLimitMiddleware(unittest.TestCase):
    def setUp(self):
        app = FastAPI()

        app.add_middleware(
            UploadBodyLimitMiddleware,
            max_body_bytes=10,
        )

        @app.post("/api/documents/upload")
        async def upload(request: Request):
            body = await request.body()
            return {"received": len(body)}

        @app.post("/api/documents/ingest")
        async def ingest(request: Request):
            body = await request.body()
            return {"received": len(body)}

        @app.post("/api/other")
        async def other(request: Request):
            body = await request.body()
            return {"received": len(body)}

        self.client = TestClient(app)

    def test_rejects_body_larger_than_limit(self):
        response = self.client.post(
            "/api/documents/upload",
            content=b"12345678901",
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json()["detail"]["code"],
            "UPLOAD_REQUEST_TOO_LARGE",
        )

    def test_accepts_body_exactly_at_limit(self):
        response = self.client.post(
            "/api/documents/upload",
            content=b"1234567890",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["received"], 10)

    def test_ingest_route_also_enforces_limit(self):
        response = self.client.post(
            "/api/documents/ingest",
            content=b"12345678901",
        )

        self.assertEqual(response.status_code, 413)

    def test_does_not_limit_unrelated_routes(self):
        response = self.client.post(
            "/api/other",
            content=b"12345678901",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["received"], 11)





class TestStreamingUploadBodyLimit(unittest.IsolatedAsyncioTestCase):
    async def _run_request(self, chunks):
        messages = iter([
            {
                "type": "http.request",
                "body": chunk,
                "more_body": index < len(chunks) - 1,
            }
            for index, chunk in enumerate(chunks)
        ])

        async def receive():
            return next(messages)

        async def endpoint(scope, receive, send):
            total = 0

            while True:
                message = await receive()

                if message["type"] == "http.request":
                    total += len(message.get("body", b""))

                    if not message.get("more_body", False):
                        break

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send({
                "type": "http.response.body",
                "body": str(total).encode(),
            })

        middleware = UploadBodyLimitMiddleware(
            endpoint,
            max_body_bytes=10,
        )

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/documents/upload",
            "headers": [],
        }

        sent = []

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)
        return sent

    async def test_streaming_body_without_content_length_is_rejected(self):
        sent = await self._run_request([
            b"123456",
            b"78901",
        ])

        self.assertEqual(sent[0]["status"], 413)

        body = json.loads(sent[1]["body"])
        self.assertEqual(
            body["detail"]["code"],
            "UPLOAD_REQUEST_TOO_LARGE",
        )

    async def test_streaming_body_exactly_at_limit_is_accepted(self):
        sent = await self._run_request([
            b"123456",
            b"7890",
        ])

        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(sent[1]["body"], b"10")



if __name__ == "__main__":
    unittest.main()
