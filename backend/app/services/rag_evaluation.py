"""Repeatable offline RAG evaluation metrics (no provider calls or DB writes).

Deterministic metrics are always computed. Optional LLM-as-judge hooks are
separated and disabled unless explicitly invoked by a caller — this module
never sends private document text to external judges by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


REQUIRED_CASE_FIELDS = ("case_id",)


def retrieval_metrics(relevant_ids: list[str], retrieved_ids: list[str], k: int = 5) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    ranked = list(retrieved_ids[: max(0, k)])
    hits = [rank for rank, item in enumerate(ranked, 1) if item in relevant]
    recall = len(hits) / len(relevant) if relevant else 1.0
    precision = len(hits) / k if k > 0 else 0.0
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def citation_metrics(
    relevant_ids: list[str],
    citation_ids: list[str],
) -> dict[str, float | None]:
    relevant = set(relevant_ids)
    cited = set(citation_ids)
    if not relevant and not cited:
        return {"citation_precision": 1.0, "citation_recall": 1.0}
    if not cited:
        return {"citation_precision": 0.0, "citation_recall": 0.0 if relevant else 1.0}
    if not relevant:
        # Citations present with no expected support — treat as incorrect.
        return {"citation_precision": 0.0, "citation_recall": 1.0}
    overlap = relevant & cited
    return {
        "citation_precision": len(overlap) / len(cited),
        "citation_recall": len(overlap) / len(relevant),
    }


def faithfulness_heuristic(answer: str, evidence_texts: list[str]) -> float | None:
    """Offline groundedness proxy: answer tokens covered by evidence tokens.

    Not an LLM judge. Returns None when there is no answer or evidence to score.
    """
    answer = (answer or "").casefold().strip()
    if not answer:
        return None
    evidence = " ".join(evidence_texts or []).casefold()
    if not evidence:
        return 0.0
    answer_tokens = {tok for tok in answer.split() if len(tok) > 2}
    if not answer_tokens:
        return 1.0
    covered = sum(1 for tok in answer_tokens if tok in evidence)
    return covered / len(answer_tokens)


def answer_relevance_heuristic(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    answer_cf = (answer or "").casefold()
    hits = sum(1 for word in keywords if str(word).casefold() in answer_cf)
    return hits / len(keywords)


def validate_golden_dataset(dataset: dict[str, Any]) -> None:
    if not isinstance(dataset, dict):
        raise ValueError("Golden dataset must be an object")
    if not isinstance(dataset.get("cases"), list):
        raise ValueError("Golden dataset must include a cases list")
    if not dataset.get("version"):
        raise ValueError("Golden dataset must declare a version")
    review = dataset.get("review")
    if not isinstance(review, dict) or review.get("status") not in {"approved", "draft"}:
        raise ValueError("Golden dataset review.status must be 'approved' or 'draft'")
    seen: set[str] = set()
    for case in dataset["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Golden dataset cases must be objects")
        for field in REQUIRED_CASE_FIELDS:
            if not case.get(field):
                raise ValueError(f"Golden case missing {field}")
        cid = str(case["case_id"])
        if cid in seen:
            raise ValueError(f"Duplicate golden case_id: {cid}")
        seen.add(cid)
        if "relevant_chunk_ids" in case and not isinstance(case["relevant_chunk_ids"], list):
            raise ValueError(f"relevant_chunk_ids must be a list for case {cid}")


def evaluate(
    dataset: dict[str, Any],
    results: dict[str, Any],
    k: int = 5,
    *,
    include_llm_judge: bool = False,
) -> dict[str, Any]:
    """Score captured RAG results. ``include_llm_judge`` is reserved and ignored offline."""
    del include_llm_judge  # Explicitly separated; never auto-invoked here.
    validate_golden_dataset(dataset)
    if not isinstance(results, dict):
        raise ValueError("Evaluation results have invalid structure")
    result_by_id = results.get("cases", {})
    if not isinstance(result_by_id, dict):
        raise ValueError("Evaluation results.cases must be an object keyed by case_id")
    expected_ids = {case.get("case_id") for case in dataset["cases"]}
    missing_ids = expected_ids - set(result_by_id)
    if missing_ids:
        raise ValueError("Evaluation results are missing cases: " + ", ".join(sorted(str(x) for x in missing_ids)))

    per_case = []
    retrieval_rows = []
    answer_matches = []
    abstention_matches = []
    citation_matches = []
    citation_precision_rows = []
    citation_recall_rows = []
    faithfulness_rows = []
    relevance_rows = []
    acl_violation_flags = []
    latencies = []
    embedding_calls = rerank_calls = provider_failures = 0

    for case in dataset["cases"]:
        cid = case["case_id"]
        observed = result_by_id.get(cid, {})
        if not isinstance(observed, dict):
            raise ValueError(f"Result for case {cid} must be an object")
        retrieved = [str(value) for value in observed.get("retrieved_chunk_ids", [])]
        expected = [str(value) for value in case.get("relevant_chunk_ids", [])]
        metrics = retrieval_metrics(expected, retrieved, k)
        if expected:
            retrieval_rows.append(metrics)
        elif not retrieved:
            # Empty expected + empty retrieved still contributes perfect empty retrieval.
            retrieval_rows.append(metrics)

        answer = str(observed.get("answer") or "")
        keywords = [str(word) for word in case.get("expected_answer_keywords", [])]
        relevance = answer_relevance_heuristic(answer, keywords)
        if relevance is not None:
            relevance_rows.append(relevance)
            answer_matches.append(float(relevance >= 1.0))

        expected_abstain = bool(case.get("expected_abstain", False))
        abstained = bool(observed.get("abstained", False))
        abstention_matches.append(float(expected_abstain == abstained))

        citation_ids = [str(value) for value in observed.get("citation_chunk_ids", [])]
        cite = citation_metrics(expected, citation_ids)
        if expected:
            citation_matches.append(float(bool(set(expected) & set(citation_ids))))
            citation_precision_rows.append(float(cite["citation_precision"] or 0.0))
            citation_recall_rows.append(float(cite["citation_recall"] or 0.0))

        evidence_texts = [str(t) for t in observed.get("evidence_texts", [])]
        faith = faithfulness_heuristic(answer, evidence_texts)
        if faith is not None and evidence_texts:
            faithfulness_rows.append(faith)

        unauthorized = observed.get("unauthorized_chunk_ids") or []
        acl_violation = bool(unauthorized) or bool(observed.get("acl_violation", False))
        if case.get("expect_acl_isolation"):
            acl_violation_flags.append(float(not acl_violation and not retrieved))
        elif "acl_violation" in observed or unauthorized:
            acl_violation_flags.append(float(not acl_violation))

        if observed.get("latency_ms") is not None:
            latencies.append(float(observed["latency_ms"]))
        embedding_calls += int(observed.get("embedding_calls", 0))
        rerank_calls += int(observed.get("rerank_calls", 0))
        provider_failures += int(observed.get("provider_failures", 0))

        per_case.append({
            "case_id": cid,
            **metrics,
            "answer_keyword_match": (relevance >= 1.0) if relevance is not None else None,
            "answer_relevance": relevance,
            "abstention_match": expected_abstain == abstained,
            "citation_match": bool(set(expected) & set(citation_ids)) if expected else None,
            "citation_precision": cite["citation_precision"],
            "citation_recall": cite["citation_recall"],
            "faithfulness": faith,
            "acl_violation": acl_violation,
        })

    metric_names = list(retrieval_rows[0]) if retrieval_rows else []
    aggregate = {name: mean(row[name] for row in retrieval_rows) for name in metric_names}
    report = {
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_review_status": (dataset.get("review") or {}).get("status"),
        "case_count": len(dataset["cases"]),
        "retrieval_metrics": aggregate,
        "answer_keyword_match_rate": mean(answer_matches) if answer_matches else None,
        "answer_relevance": mean(relevance_rows) if relevance_rows else None,
        "citation_match_rate": mean(citation_matches) if citation_matches else None,
        "citation_precision": mean(citation_precision_rows) if citation_precision_rows else None,
        "citation_recall": mean(citation_recall_rows) if citation_recall_rows else None,
        "faithfulness": mean(faithfulness_rows) if faithfulness_rows else None,
        "abstention_accuracy": mean(abstention_matches) if abstention_matches else None,
        "acl_isolation_accuracy": mean(acl_violation_flags) if acl_violation_flags else None,
        "mean_latency_ms": mean(latencies) if latencies else None,
        "embedding_calls": embedding_calls,
        "rerank_calls": rerank_calls,
        "provider_failures": provider_failures,
        "llm_judge": {"enabled": False, "scores": None},
        "per_case": per_case,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score captured RAG results without contacting providers or databases."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True, help="JSON keyed by case_id under `cases`")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", help="Optional path to write the JSON report")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = evaluate(dataset, results, args.k)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
