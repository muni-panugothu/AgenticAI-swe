# Memory Security (Topics 20–21)

## Scope semantics

| Scope | Storage | Enforcement |
|-------|---------|-------------|
| **User** | `user_id`, `owner_id`, optional null `tenant_id`/`project_id` | Default; Mongo filters + permission checks |
| **Tenant** | `tenant_id` on document | Caller `tenant_id` must match; mismatches deny |
| **Project** | `project_id` + `scope=project` | Active project from Redis session; cross-project deny |
| **Conversation** | Optional `conversation_id` on long-term rows | Without matching `conversation_id`, row is **hidden** from retrieval |
| **Session chat** | Redis / `temporary_memory_service` | Not stored in `memories_history`; separate TTL |

### Missing metadata (safe defaults)

- **No owner / user_id:** deny read/write; never `public_read`.
- **No tenant on doc when caller has tenant:** legacy user-level only when both sides null-tenant.
- **Project-tagged doc + unscoped query:** deny.
- **Orphan docs:** use `memory_legacy_service.audit_orphan_memory_counts()`; do not auto-assign owners.

## Permissions

Central service: `memory_permission_service.py`

- **Owner:** read/write/delete in matching tenant/project.
- **shared_with:** read-only.
- **project_read / project_write:** same project context only.
- **public_read:** same non-null tenant only.
- Denials audited via `permission_denied` (no content in logs).

## Semantic retrieval

Pipeline: Atlas `$vectorSearch` pre-filter (`user_id`, optional `tenant_id`/`project_id`) → `document_matches_scope` → `check_read_permission` → `filter_readable_documents`.

Index `memory_vector_index` does not filter `shared_with`; post-filters are mandatory.

## API identity

Memory mutations run through authenticated `chat_with_ai` tools; identity is JWT `sub` + `tenant_id`. HTTP bodies must not send `user_id`, `owner_id`, `tenant_id`, `project_id`, or `conversation_id` (`reject_untrusted_user_id_in_body`).

There is no public REST memory CRUD; security tests hit the service layer and `/api/chat`.

## Legacy migration

1. Run `audit_orphan_memory_counts` (read-only).
2. Manually repair or quarantine IDs in `sample_ids_missing_owner`.
3. Never bulk-assign orphans to an active user.

## Tests

```powershell
cd backend
python -m unittest tests.test_memory_security tests.test_memory_scope_permission_security -v
python -m unittest discover -s tests
```
