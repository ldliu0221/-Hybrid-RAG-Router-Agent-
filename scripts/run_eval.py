import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

from app.services.eval_service import EvalService


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "app" / "eval" / "dataset.json"
RESULTS_PATH = ROOT_DIR / "app" / "eval" / "results.json"


def main():
    eval_service = EvalService()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"数据集文件不存在: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    route_buckets = defaultdict(list)

    print("=== Running Evaluation ===")
    print(f"Dataset path: {DATASET_PATH}\n")

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        result = eval_service.run_single_eval(
            question=question,
            ground_truth=ground_truth,
            use_agent=True,
            top_k=5,
        )

        results.append(result)
        route_buckets[result["route"]].append(result["score"])

        print(f"[{i}] Question: {question}")
        print(f"Route: {result['route']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Score: {result['score']}")
        print(f"Reason: {result['reason']}")
        print("-" * 60)

    avg_score = mean([r["score"] for r in results]) if results else 0.0

    summary = {
        "total_samples": len(results),
        "average_score": round(avg_score, 4),
        "route_stats": {
            route: {
                "count": len(scores),
                "average_score": round(mean(scores), 4) if scores else 0.0
            }
            for route, scores in route_buckets.items()
        }
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = {
        "summary": summary,
        "results": results
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()