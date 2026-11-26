#!/usr/bin/env python3
"""CLI for the Phase A assistant evaluation framework.

Default: offline (mocked providers / no live API calls).
Live mode: pass --live AND set ASSISTANT_EVAL_LIVE=1.

Examples:
  python -m scripts.run_assistant_eval
  python -m scripts.run_assistant_eval --output reports/assistant_eval_latest.json
  ASSISTANT_EVAL_LIVE=1 python -m scripts.run_assistant_eval --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path when invoked as a file.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.eval.assistant_eval.loader import category_counts, default_dataset_path, load_dataset
from app.eval.assistant_eval.runner import run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run assistant_eval_v1 evaluation suite")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(default_dataset_path()),
        help="Path to dataset.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live provider evaluation (also requires ASSISTANT_EVAL_LIVE=1)",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Limit to category (repeatable)",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Limit to case_id (repeatable)",
    )
    parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit non-zero if any case status is FAIL",
    )
    args = parser.parse_args(argv)

    if args.live and os.environ.get("ASSISTANT_EVAL_LIVE", "").strip() not in {
        "1",
        "true",
        "True",
        "yes",
    }:
        print(
            "Live mode refused: set ASSISTANT_EVAL_LIVE=1 to allow expensive provider calls.",
            file=sys.stderr,
        )
        return 2

    dataset = load_dataset(args.dataset)
    report = run_evaluation(
        dataset=dataset,
        live=bool(args.live),
        categories=args.category or None,
        case_ids=args.case_id or None,
    )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote report: {out_path}")
    else:
        print(text)

    summary = report.get("summary") or {}
    print(
        f"dataset={report.get('dataset_version')} mode={report.get('mode')} "
        f"PASS={summary.get('PASS', 0)} FAIL={summary.get('FAIL', 0)} "
        f"BLOCKED={summary.get('BLOCKED', 0)} NEEDS_REVIEW={summary.get('NEEDS_REVIEW', 0)} "
        f"categories={category_counts(dataset)}",
        file=sys.stderr,
    )

    if args.fail_on_fail and summary.get("FAIL", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
