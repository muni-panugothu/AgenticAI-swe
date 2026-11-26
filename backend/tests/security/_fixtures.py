"""Shared offline fixtures for Batch D adversarial tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.services import approval_service, tool_permission_service
from app.services.security_store import InMemoryCollection

_ENV_KEYS = (
    "AGENT_HITL_ENABLED",
    "AGENT_TOOL_PERMISSIONS_ENFORCE",
    "AGENT_SECURITY_POLICY_VERSION",
    "AGENT_PERMISSION_ADMIN_USER_IDS",
    "JWT_SECRET",
)


class SecurityCase(unittest.TestCase):
    """HITL + permissions enforced against isolated in-memory stores."""

    @classmethod
    def setUpClass(cls):
        cls._env_backup = {k: os.environ.get(k) for k in _ENV_KEYS}
        os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")
        os.environ["AGENT_HITL_ENABLED"] = "true"
        os.environ["AGENT_TOOL_PERMISSIONS_ENFORCE"] = "true"
        os.environ["AGENT_SECURITY_POLICY_VERSION"] = "1"

    @classmethod
    def tearDownClass(cls):
        for key, value in cls._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def setUp(self):
        self.approvals = InMemoryCollection()
        self.perms = InMemoryCollection()
        approval_service.configure_approval_collection(self.approvals)
        tool_permission_service.configure_permission_collection(self.perms)
        tool_permission_service.invalidate_permission_cache()
        os.environ["AGENT_HITL_ENABLED"] = "true"
        os.environ["AGENT_TOOL_PERMISSIONS_ENFORCE"] = "true"
        os.environ["AGENT_SECURITY_POLICY_VERSION"] = "1"
        self._hitl_patch = patch(
            "app.services.agent_security_policy.hitl_enabled",
            return_value=True,
        )
        self._hitl_patch.start()

    def tearDown(self):
        self._hitl_patch.stop()
        approval_service.configure_approval_collection(None)
        tool_permission_service.configure_permission_collection(None)
        tool_permission_service.invalidate_permission_cache()

    def create_and_approve(
        self,
        *,
        user_id: str = "user-a",
        tool_name: str = "save_memory",
        arguments: dict | None = None,
        project_id: str | None = "proj-1",
        tenant_id: str | None = "ten-1",
    ) -> tuple[str, dict]:
        args = arguments if arguments is not None else {"memory": "secret fact"}
        created = approval_service.create_approval_request(
            user_id=user_id,
            tool_name=tool_name,
            arguments=args,
            project_id=project_id,
            tenant_id=tenant_id,
        )
        self.assertTrue(created.get("success"), created)
        aid = created["approval"]["approval_id"]
        decided = approval_service.decide_approval(
            approval_id=aid,
            reviewer_user_id=user_id,
            approve=True,
        )
        self.assertTrue(decided.get("success"), decided)
        return aid, {
            "user_id": user_id,
            "tool_name": tool_name,
            "arguments": args,
            "project_id": project_id,
            "tenant_id": tenant_id,
        }
