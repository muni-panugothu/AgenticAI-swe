"""Isolated Mongo integration coverage for cross-process refresh rotation.

Set AUTH_TEST_MONGO_URI to an explicitly local MongoDB URI to enable these
tests. A random, disposable database is used and dropped after the test class.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from pymongo import MongoClient

_URI = os.getenv("AUTH_TEST_MONGO_URI")
_LOCAL_URI = bool(_URI and ("127.0.0.1" in _URI or "localhost" in _URI))

os.environ.setdefault("JWT_SECRET", "isolated-auth-refresh-test-secret-32chars")

from app.core.config import get_settings  # noqa: E402
from app.core.security import decode_access_token  # noqa: E402
from app.services import auth_service  # noqa: E402


@unittest.skipUnless(_LOCAL_URI, "requires explicit local AUTH_TEST_MONGO_URI")
class TestMongoRefreshIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_name = f"auth_refresh_test_{uuid.uuid4().hex}"
        cls.client = MongoClient(_URI, serverSelectionTimeoutMS=3000)
        cls.client.admin.command("ping")
        cls.db = cls.client[cls.db_name]

    @classmethod
    def tearDownClass(cls):
        cls.client.drop_database(cls.db_name)
        cls.client.close()

    def setUp(self):
        self.db.drop_collection("users")
        self.db.drop_collection("sessions")
        self.users = self.db["users"]
        self.sessions = self.db["sessions"]
        self._env = patch.dict(
            os.environ,
            {
                "JWT_SECRET": "isolated-auth-refresh-test-secret-32chars",
                "MEMORY_DATABASE": self.db_name,
                "AUTH_USERS_COLLECTION": "users",
                "AUTH_SESSIONS_COLLECTION": "sessions",
            },
        )
        self._env.start()
        get_settings.cache_clear()
        self.addCleanup(self._restore_settings)
        self.users.insert_one(
            {
                "_id": f"usr_{uuid.uuid4().hex}",
                "email": f"mongo-{uuid.uuid4().hex}@example.com",
                "status": "active",
                "tenant_id": f"ten_{uuid.uuid4().hex[:12]}",
            }
        )
        self.user = self.users.find_one({})
        self._users_patch = patch.object(auth_service, "_users_collection", lambda: self.users)
        self._sessions_patch = patch.object(auth_service, "_sessions_collection", lambda: self.sessions)
        self._users_patch.start()
        self._sessions_patch.start()
        self.addCleanup(self._sessions_patch.stop)
        self.addCleanup(self._users_patch.stop)

    def _restore_settings(self):
        get_settings.cache_clear()
        self._env.stop()

    def _new_session(self):
        return auth_service.issue_tokens_for_user(self.user)

    def _child_env(self, **values):
        env = os.environ.copy()
        env.update(
            {
                "JWT_SECRET": "isolated-auth-refresh-test-secret-32chars",
                "MONGO_URI": _URI,
                "MEMORY_DATABASE": self.db_name,
                "AUTH_USERS_COLLECTION": "users",
                "AUTH_SESSIONS_COLLECTION": "sessions",
            }
        )
        env.update(values)
        return env

    def test_reuse_revokes_only_family_and_other_users_sessions_survive(self):
        first = self._new_session()
        other_device = self._new_session()
        unrelated_user = {
            "_id": f"usr_{uuid.uuid4().hex}",
            "email": f"other-{uuid.uuid4().hex}@example.com",
            "status": "active",
        }
        self.users.insert_one(unrelated_user)
        other_user_session = auth_service.issue_tokens_for_user(unrelated_user)

        rotated = auth_service.refresh_session(refresh_token=first["refresh_token"])
        self.assertTrue(rotated["success"])
        self.assertNotEqual(first["refresh_token"], rotated["refresh_token"])
        second_rotation = auth_service.refresh_session(
            refresh_token=rotated["refresh_token"]
        )
        self.assertTrue(second_rotation["success"])
        self.assertNotEqual(rotated["refresh_token"], second_rotation["refresh_token"])
        self.sessions.update_one(
            {"token_hash": auth_service.hash_refresh_token(first["refresh_token"])},
            {"$set": {"rotated_at": datetime.now(timezone.utc) - timedelta(seconds=10)}},
        )
        self.assertIsNotNone(self.sessions.find_one({"token_hash": auth_service.hash_refresh_token(first["refresh_token"])})["revoked_at"])

        replay = auth_service.refresh_session(refresh_token=first["refresh_token"])
        self.assertFalse(replay["success"])
        self.assertEqual(replay["code"], "REFRESH_REUSE")
        root_row = self.sessions.find_one(
            {"token_hash": auth_service.hash_refresh_token(first["refresh_token"])}
        )
        self.assertIsNotNone(root_row.get("reuse_detected_at"))
        self.assertGreaterEqual(root_row.get("reuse_count", 0), 1)
        rotated_claims = decode_access_token(second_rotation["access_token"])
        other_device_claims = decode_access_token(other_device["access_token"])
        unrelated_claims = decode_access_token(other_user_session["access_token"])
        self.assertFalse(auth_service.get_active_session(rotated_claims["sid"], self.user["_id"]))
        self.assertTrue(auth_service.get_active_session(other_device_claims["sid"], self.user["_id"]))
        self.assertTrue(auth_service.get_active_session(unrelated_claims["sid"], unrelated_user["_id"]))

    def test_expired_malformed_and_unknown_refresh_values_are_rejected(self):
        expired = "synthetic-expired-refresh-value"
        self.sessions.insert_one(
            {
                "_id": f"sess_{uuid.uuid4().hex}",
                "user_id": self.user["_id"],
                "token_hash": auth_service.hash_refresh_token(expired),
                "created_at": datetime.now(timezone.utc) - timedelta(days=2),
                "expires_at": datetime.now(timezone.utc) - timedelta(days=1),
                "revoked_at": None,
                "family_id": f"family_{uuid.uuid4().hex}",
            }
        )
        self.assertEqual(
            auth_service.refresh_session(refresh_token=expired)["code"],
            "EXPIRED_REFRESH",
        )
        # Refresh credentials are opaque random values, not JWTs; a bad JWT
        # signature is therefore not a meaningful refresh-token case. Any
        # malformed/forged opaque value must simply miss the stored hash.
        for malformed in ("", "not-a-token", "eyJhbGciOiJub25lIn0.invalid.sig"):
            with self.subTest(value_class="empty" if not malformed else "forged"):
                result = auth_service.refresh_session(refresh_token=malformed)
                self.assertFalse(result["success"])
                self.assertEqual(result["code"], "INVALID_REFRESH")

    def test_two_processes_race_atomically_and_unrelated_session_survives(self):
        original = self._new_session()
        unrelated = self._new_session()
        script = r'''
import json, os, time
from pymongo import MongoClient
from app.services.auth_service import refresh_session
client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=3000)
while time.time() < float(os.environ["REFRESH_AT"]):
    time.sleep(0.002)
result = refresh_session(refresh_token=os.environ["TEST_REFRESH_TOKEN"])
print(json.dumps({"success": bool(result.get("success")), "code": result.get("code")}))
client.close()
'''
        start_at = time.time() + 2.0
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=self._child_env(
                    TEST_REFRESH_TOKEN=original["refresh_token"],
                    REFRESH_AT=str(start_at),
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            outcomes.append(json.loads(stdout.strip()))

        self.assertEqual(sum(outcome["success"] for outcome in outcomes), 1, outcomes)
        self.assertEqual(sum(outcome.get("code") == "REFRESH_RACE" for outcome in outcomes), 1, outcomes)
        unrelated_claims = decode_access_token(unrelated["access_token"])
        self.assertTrue(auth_service.get_active_session(unrelated_claims["sid"], self.user["_id"]))
        root = self.sessions.find_one({"user_id": self.user["_id"], "rotated_from": None})
        self.assertIsNotNone(root["revoked_at"])
        lineage = list(self.sessions.find({"family_id": root["family_id"]}))
        self.assertGreaterEqual(len(lineage), 2)
        self.assertIsNotNone(root.get("last_concurrent_refresh_at"))
        self.assertEqual(sum(row["revoked_at"] is None for row in lineage), 1)

        # Once the concurrency grace window closes, the old credential is
        # treated as reuse and invalidates the compromised lineage.
        time.sleep(auth_service._REFRESH_RACE_GRACE_SECONDS + 0.1)
        replay_script = r'''
import json, os
from app.services.auth_service import refresh_session
result = refresh_session(refresh_token=os.environ["TEST_REFRESH_TOKEN"])
print(json.dumps({"success": bool(result.get("success")), "code": result.get("code")}))
'''
        replay = subprocess.run(
            [sys.executable, "-c", replay_script],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=self._child_env(TEST_REFRESH_TOKEN=original["refresh_token"]),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout.strip())["code"], "REFRESH_REUSE")
        family_after_reuse = list(self.sessions.find({"family_id": root["family_id"]}))
        self.assertTrue(all(row["revoked_at"] is not None for row in family_after_reuse))
        root_after_reuse = self.sessions.find_one({"_id": root["_id"]})
        self.assertGreaterEqual(root_after_reuse.get("reuse_count", 0), 1)

    def test_revocation_persists_and_is_rejected_in_another_process(self):
        issued = self._new_session()
        claims = decode_access_token(issued["access_token"])
        self.assertTrue(auth_service.revoke_refresh_token(issued["refresh_token"]))
        self.assertFalse(auth_service.get_active_session(claims["sid"], self.user["_id"]))
        script = r'''
import json, os
from app.services.auth_service import refresh_session
result = refresh_session(refresh_token=os.environ["TEST_REFRESH_TOKEN"])
print(json.dumps({"success": bool(result.get("success")), "code": result.get("code")}))
'''
        child = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=self._child_env(TEST_REFRESH_TOKEN=issued["refresh_token"]),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout.strip()), {"success": False, "code": "REFRESH_REUSE"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
