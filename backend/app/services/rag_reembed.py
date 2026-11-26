"""Controlled CLI for re-embedding a single owned RAG document.

Never exposes an unauthenticated HTTP endpoint. Defaults to dry-run.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed one owned document's chunks when embedding metadata is stale. "
            "Requires explicit --apply to write. Does not bulk-migrate production."
        )
    )
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed even if metadata matches")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform writes. Without this flag, only a dry-run report is produced.",
    )
    args = parser.parse_args()

    # Safety: refuse accidental production DB unless operator opts in.
    if args.apply and os.getenv("RAG_REEMBED_CONFIRM", "").lower() not in {"1", "true", "yes"}:
        raise SystemExit(
            "Refusing to apply re-embed without RAG_REEMBED_CONFIRM=1 "
            "(and --apply). Use dry-run first."
        )

    from app.services.rag_document_service import reembed_document_chunks

    result = reembed_document_chunks(
        document_id=args.document_id,
        owner_id=args.owner_id,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
        force=bool(args.force),
        dry_run=not bool(args.apply),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
