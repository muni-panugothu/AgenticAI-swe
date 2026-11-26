# Golden v2 ↔ corpus mapping

Synthetic evaluation-only content. Not production data.

## ID strategy

**Direct fixture insertion** with exact golden IDs (`doc-policy-md:2`, etc.).

Normal `ingest_document()` hashing is **not** used for capture seeding. Runtime ACL
(`_same_scope`, `authorize_candidates`) is unchanged and still enforced on retrieve.

Authoritative chunk texts and indexes live in `manifest.json`. Source files are
human-readable mirrors of those facts.

## Documents

| document_id | owner | tenant | project | share | Relevant facts |
|---|---|---|---|---|---|
| `doc-policy-md` | alice | t1 | null | private | Chunk `:2` — notice period **14 days** |
| `doc-handbook-pdf` | bob | t1 | null | shared_read → alice | Chunk `:1` — drills **quarterly** |
| `doc-roster-csv` | alice | t1 | proj-hr | private | Chunk `:0` — **3** engineers |
| `doc-other-tenant` | mallory | t2 | null | private | Isolation negative; not for alice/t1 |

## Cases → ACL scope (capture principals)

| case_id | user_id | tenant_id | project_id | Notes |
|---|---|---|---|---|
| md-leave-notice | alice | t1 | null | Owner of policy |
| pdf-handbook-safety | alice | t1 | null | Shared handbook |
| csv-roster-count | alice | t1 | proj-hr | From metadata_filters |
| multi-doc-policy-and-handbook | alice | t1 | null | Policy + shared handbook |
| no-support-home-address | alice | t1 | null | Expect abstain |
| acl-other-tenant | alice | t1 | null | Must not see mallory/t2 |
| acl-private-owner | alice | t1 | null | Own policy |
| acl-shared-read | alice | t1 | null | Shared handbook |
| metadata-project-filter | alice | t1 | proj-hr | Roster project filter |

## Assumptions

1. Golden labels already encode intended chunk indexes; corpus texts are written so those indexes hold the required keywords.
2. `file_type: pdf` is metadata only; seeded text comes from `handbook.txt` (no binary PDF required for vector/overlap retrieval of fixture text).
3. Retrieval-only captures leave `answer` empty; `abstained` means zero packed candidates.
