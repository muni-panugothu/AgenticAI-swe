# Topics 48–53 — Agent Security Architecture

## Enforcement boundary

All tool calls pass through `prepare_tool_invocation` → `execute_tool`.
Permissions, approval, provenance stripping, and sandbox gates run there.
Reliability wrappers (Topics 45–47), parallel dispatch, and native async paths
do not bypass this boundary.

## Topic 48 — Human-in-the-loop

- Enabled only when `AGENT_HITL_ENABLED=true` (default **false** so Topics 1–47 stay unchanged).
- Server policy (`agent_security_policy.get_tool_risk_meta`) marks high-risk tools.
- LLM cannot exempt a tool from approval.
- Pending approval returns `success=false`, `approval_required=true`, `execution_completed=false`.

## Topic 49 — Tool permissions

- Deny by default for unknown tools.
- Baseline authenticated catalog for known chat tools.
- Explicit Mongo DENY overrides ALLOW.
- Users cannot self-grant via public API (`AGENT_PERMISSION_ADMIN_USER_IDS` required).
- Short TTL cache invalidated on revoke.

## Topic 50 — Approvals

- Bound to user, project, tenant, tool, argument digest, policy version.
- States: pending → approved/rejected/expired/revoked/cancelled; approved → consumed; consumed → execution_failed.
- Atomic `find_one_and_update` consumption (one-time).
- Approval never overrides a permission denial.
- REST: `/api/approvals/{id}`, `/decision`, `/revoke`, `/execute`.

## Topic 51 — Prompt injection

- Trust levels in `ContentTrustLevel`.
- Untrusted identity/security fields stripped from tool arguments.
- RAG/memory labeled untrusted; cannot authorize.
- Enforcement is backend rejection, not prompt text alone.

## Topic 52 — Tool security

- Identity fields from LLM rejected.
- Scope injected from authenticated context.
- Security audit events (no secrets).
- Authorization failures classified so retries/fallbacks cannot bypass.

## Topic 53 — Sandboxing

**Active:** path confinement (`path_sandbox`), sensitive deny list, file size limits, tool timeouts, empty process allowlist (no shell tool).

**Not enforced here:** OS containers, Windows Job Objects, network namespaces, escape-proof isolation. Junction/reparse races remain a residual Windows risk. Do not advertise escape-proof sandboxing.

## Threat model

See `docs/agent_security_threat_model.md`.
