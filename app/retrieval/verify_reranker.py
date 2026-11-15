"""Check that the cross-encoder downloads, loads and ranks sensibly.

Usage:
    python -m app.retrieval.verify_reranker
"""

import time

from app.retrieval.reranker import get_reranker

QUERY = "online store with shopping cart and card payments"
DOCUMENTS = [
    "Shopping cart and checkout flow with card payments for a headless e-commerce storefront.",
    "Predictive maintenance models trained on factory IoT vibration telemetry.",
]


def main() -> None:
    reranker = get_reranker()
    started_at = time.perf_counter()
    scores = reranker.score(QUERY, DOCUMENTS)
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    print(f"model: {reranker.model_name}")
    for score, document in zip(scores, DOCUMENTS):
        print(f"{score:8.3f}  {document}")
    print(f"scoring time: {elapsed_ms:.1f} ms for {len(DOCUMENTS)} pairs")

    if scores[0] <= scores[1]:
        raise SystemExit("FAIL: the relevant document did not score higher")
    print("OK: reranker loaded and ranks the relevant document first")


if __name__ == "__main__":
    main()
