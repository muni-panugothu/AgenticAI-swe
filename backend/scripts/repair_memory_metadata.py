"""
Optional metadata backfill for exclusive memory facts.

Dry-run by default. Fills missing fact_key / normalized_value from
deterministic extraction and propagates conflict_group_id from pending
rows onto linked active partners. Never invents groups from embeddings.
Never changes status, memory text, or embeddings.

Usage:
    python -m scripts.repair_memory_metadata --user-id muni
    python -m scripts.repair_memory_metadata --user-id muni --fact-key preferred_database
    python -m scripts.repair_memory_metadata --user-id muni --apply
"""

from __future__ import annotations

import argparse
import json

from app.services.memory_service import (
    backfill_memory_metadata,
    inspect_memory_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional read-repair for memory metadata (dry-run default)."
    )
    parser.add_argument("--user-id", required=True, help="Authenticated user id")
    parser.add_argument(
        "--fact-key",
        default=None,
        help="Optional canonical fact key filter (e.g. preferred_database)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply proposed patches. Without this flag, dry-run only.",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"=== Memory metadata backfill ({mode}) ===")
    print(f"user_id={args.user_id} fact_key={args.fact_key}")

    before = None
    if args.fact_key:
        before = inspect_memory_metadata(
            args.user_id,
            fact_key=args.fact_key,
        )
        print("\n-- Before inspection --")
        print(
            json.dumps(
                before.get("conflict_group_comparison"),
                indent=2,
                ensure_ascii=False,
            )
        )

    result = backfill_memory_metadata(
        args.user_id,
        fact_key=args.fact_key,
        dry_run=dry_run,
        limit=args.limit,
    )
    print("\n-- Backfill result --")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    if args.fact_key:
        after = inspect_memory_metadata(
            args.user_id,
            fact_key=args.fact_key,
        )
        print("\n-- After inspection --")
        print(
            json.dumps(
                {
                    "records": after.get("records"),
                    "conflict_group_comparison": after.get(
                        "conflict_group_comparison"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    if dry_run:
        print(
            "\nDry-run complete. Re-run with --apply to write proposed fields."
        )


if __name__ == "__main__":
    main()
