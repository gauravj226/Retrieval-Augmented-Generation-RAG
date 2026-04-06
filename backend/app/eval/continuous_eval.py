import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_cases(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_retrieval(case: Dict) -> float:
    gold = set((case.get("expected_sources") or []))
    got = set((case.get("sources") or []))
    if not gold:
        return 1.0
    return len(gold.intersection(got)) / len(gold)


def score_grounding(case: Dict) -> float:
    answer = (case.get("answer") or "").lower()
    snippets = " ".join(case.get("contexts") or []).lower()
    if not answer:
        return 0.0
    overlap = [w for w in answer.split() if len(w) > 4 and w in snippets]
    return min(1.0, len(overlap) / 20.0)


def score_relevance(case: Dict) -> float:
    q = set((case.get("question") or "").lower().split())
    a = set((case.get("answer") or "").lower().split())
    if not q:
        return 0.0
    return len(q.intersection(a)) / len(q)


def run(dataset: Path, min_retrieval: float, min_grounding: float, min_relevance: float) -> int:
    cases = load_cases(dataset)
    if not cases:
        print("No eval cases found.")
        return 1
    retrieval = sum(score_retrieval(c) for c in cases) / len(cases)
    grounding = sum(score_grounding(c) for c in cases) / len(cases)
    relevance = sum(score_relevance(c) for c in cases) / len(cases)

    print(f"retrieval_hit_rate={retrieval:.3f}")
    print(f"grounding_proxy={grounding:.3f}")
    print(f"answer_relevance={relevance:.3f}")

    if retrieval < min_retrieval or grounding < min_grounding or relevance < min_relevance:
        print("Evaluation thresholds not met.")
        return 2
    print("Evaluation thresholds met.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous RAG evaluation harness.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset")
    parser.add_argument("--min-retrieval", type=float, default=0.65)
    parser.add_argument("--min-grounding", type=float, default=0.45)
    parser.add_argument("--min-relevance", type=float, default=0.45)
    args = parser.parse_args()
    return run(
        dataset=Path(args.dataset),
        min_retrieval=args.min_retrieval,
        min_grounding=args.min_grounding,
        min_relevance=args.min_relevance,
    )


if __name__ == "__main__":
    sys.exit(main())
