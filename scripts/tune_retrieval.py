import json
from pathlib import Path
from statistics import mean

from app.services.eval_service import EvalService


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "app" / "eval" / "dataset.json"
TUNING_RESULTS_PATH = ROOT_DIR / "app" / "eval" / "tuning_results.json"


def run_eval_once(config: dict):
    eval_service = EvalService()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    scores = []
    for item in dataset:
        result = eval_service.run_single_eval(
            question=item["question"],
            ground_truth=item["ground_truth"],
            use_agent=True,
            top_k=config["top_k"],
            dense_weight=config["dense_weight"],
            sparse_weight=config["sparse_weight"],
            enable_rewrite=config["enable_rewrite"],
            enable_rerank=config["enable_rerank"],
        )
        scores.append(result["score"])

    return mean(scores) if scores else 0.0


def main():
    candidates = [
        {"top_k": 5, "dense_weight": 0.7, "sparse_weight": 0.3, "enable_rewrite": True, "enable_rerank": True},
        {"top_k": 5, "dense_weight": 0.8, "sparse_weight": 0.2, "enable_rewrite": True, "enable_rerank": True},
        {"top_k": 5, "dense_weight": 0.6, "sparse_weight": 0.4, "enable_rewrite": True, "enable_rerank": True},
        {"top_k": 3, "dense_weight": 0.7, "sparse_weight": 0.3, "enable_rewrite": True, "enable_rerank": True},
        {"top_k": 5, "dense_weight": 0.7, "sparse_weight": 0.3, "enable_rewrite": False, "enable_rerank": True},
        {"top_k": 5, "dense_weight": 0.7, "sparse_weight": 0.3, "enable_rewrite": True, "enable_rerank": False}
    ]

    all_results = []

    for config in candidates:
        score = run_eval_once(config)
        row = {
            **config,
            "average_score": round(score, 4)
        }
        all_results.append(row)
        print(row)

    best = max(all_results, key=lambda x: x["average_score"])

    print("\n=== Best Config ===")
    print(best)

    with open(TUNING_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": best,
                "all_results": all_results
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved to: {TUNING_RESULTS_PATH}")


if __name__ == "__main__":
    main()