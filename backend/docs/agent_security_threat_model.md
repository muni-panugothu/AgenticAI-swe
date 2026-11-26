# Topics 48–53 — Threat Model and Architecture Assumptions

## Trusted execution boundaries (existing)

1. JWT authentication (`api/deps`, `core/security`)
2. `prepare_tool_invocation` / `execute_tool` — validation, strip LLM identity, inject scope
3. Intent tool allowlist (`INTENT_TOOL_ALLOWLIST`)
4. Filesystem confinement (`tools/path_sandbox.py`)
5. Memory ACL (`memory_permission_service`) and RAG ACL (`rag_acl`)
6. Topics 45–47 reliability wrappers (do not bypass execute_tool)

## Gaps addressed by Topics 48–53

- Human approval before mutating / high-risk tools (`AGENT_HITL_ENABLED`, default off)
- Durable tool-permission grants/denies (Mongo) + deny-by-default outside baseline
- Atomic one-time approval consumption (`find_one_and_update`)
- Untrusted content trust labels + strip of security fields at execution boundary
- Sandbox gate: path confinement + empty process allowlist (**no shell tool**)

## Windows / runtime capabilities

| Control | Status |
|---|---|
| Path resolve + relative_to root | Enforced |
| Sensitive basename deny | Enforced |
| POSIX `O_NOFOLLOW` | Not available on typical Windows |
| Junction/reparse races | Residual risk (documented) |
| OS container / job-object isolation | **Not claimed** in this environment |
| Network isolation for tools | App-level only; web_search is explicit allow |
| Arbitrary process execution | **Not offered** (fail closed) |

## Threat coverage (summary)

Malicious user input, prompt injection via RAG/files/web/tool output, unauthorized tools, cross-user/project access, approval forgery/replay/double-consume, path traversal/symlink escape attempts, resource exhaustion via tool output limits, cancellation races. LLM decisions never authorize.

## Security assumption

Fail closed when approval/permission state cannot be verified. Do not advertise escape-proof OS isolation.
