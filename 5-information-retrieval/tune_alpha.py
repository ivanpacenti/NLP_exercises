import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "app.py"
UI_PATH = BASE_DIR / "information-retrieval-ui.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_mrr(app_module, test_data: List[Dict[str, Any]], mode: str, alpha: float, top_k: int) -> float:
    rrs: List[float] = []

    for item in test_data:
        query = item["query"]
        relevant = {r["course_id"] for r in item["relevant"]}

        ranked = app_module.rank_courses_for_query(query=query, top_k=top_k, mode=mode, alpha=alpha)
        returned = [cid for cid, _ in ranked]

        rank = None
        for i, cid in enumerate(returned, start=1):
            if cid in relevant:
                rank = i
                break

        rr = 0.0 if rank is None else 1.0 / rank
        rrs.append(rr)

    return sum(rrs) / len(rrs) if rrs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search MRR for mode/alpha.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    app_module = _load_module(APP_PATH, "ir_app")
    ui_module = _load_module(UI_PATH, "ir_ui")

    app_module._build_indexes()
    test_data = ui_module.TEST_DATA

    candidates: List[Tuple[str, float, float]] = []

    candidates.append(("sparse", 0.0, evaluate_mrr(app_module, test_data, "sparse", 0.0, args.top_k)))
    candidates.append(("dense", 1.0, evaluate_mrr(app_module, test_data, "dense", 1.0, args.top_k)))

    alpha = 0.0
    while alpha <= 1.000001:
        mrr = evaluate_mrr(app_module, test_data, "hybrid", alpha, args.top_k)
        candidates.append(("hybrid", round(alpha, 4), mrr))
        alpha += args.step

    candidates.sort(key=lambda x: x[2], reverse=True)
    best_mode, best_alpha, best_mrr = candidates[0]

    print("Top configurations by MRR:")
    for mode, alp, mrr in candidates[:10]:
        print(f"  mode={mode:6s} alpha={alp:>4} mrr={mrr:.4f}")

    print("\nBest:")
    print(f"  mode={best_mode}")
    print(f"  alpha={best_alpha}")
    print(f"  mrr={best_mrr:.4f}")


if __name__ == "__main__":
    main()
