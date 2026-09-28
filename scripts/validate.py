import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retriever import TfidfRetriever
from src.text import make_exact_query_key, normalize_text


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped offline validation")
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--max-items", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def recall_at_k(predictions, targets):
    recalls = []
    for predicted, relevant in zip(predictions, targets):
        recalls.append(len(set(predicted) & relevant) / len(relevant))
    return float(np.mean(recalls))


def main():
    args = parse_args()
    train = pd.read_parquet(args.train)
    signature_columns = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    train = train.drop_duplicates(["item_id", *signature_columns])
    train["validation_group"] = train["search_query"].map(normalize_text)
    train["target_group"] = train.apply(make_exact_query_key, axis=1)

    rng = np.random.default_rng(args.seed)
    groups = train["validation_group"].drop_duplicates().to_numpy()
    rng.shuffle(groups)
    validation_groups = set(groups[: args.max_queries])

    validation_rows = train[train["validation_group"].isin(validation_groups)]
    history_rows = train[~train["validation_group"].isin(validation_groups)]

    relevant_ids = set(validation_rows["item_id"].astype(str))
    item_rows = train.drop_duplicates("item_id")
    remaining_items = item_rows[~item_rows["item_id"].astype(str).isin(relevant_ids)]
    sample_size = max(0, min(args.max_items - len(relevant_ids), len(remaining_items)))
    sampled_items = remaining_items.sample(sample_size, random_state=args.seed)
    items = pd.concat(
        [item_rows[item_rows["item_id"].astype(str).isin(relevant_ids)], sampled_items],
        ignore_index=True,
    )

    queries = validation_rows.drop_duplicates("target_group").reset_index(drop=True)
    targets_by_group = (
        validation_rows.assign(item_id=validation_rows["item_id"].astype(str))
        .groupby("target_group")["item_id"]
        .agg(lambda values: set(values))
    )
    targets = [targets_by_group[group] for group in queries["target_group"]]

    print(f"Queries: {len(queries)}")
    print(f"Candidate items: {len(items)}")

    baseline = TfidfRetriever(enriched_text=False).fit(items, history_rows)
    baseline_predictions = baseline.predict(queries, top_k=50, use_history=False)
    print(f"Baseline title TF-IDF Recall@50: {recall_at_k(baseline_predictions, targets):.5f}")

    enriched = TfidfRetriever(enriched_text=True).fit(items, history_rows)
    text_predictions = enriched.predict(queries, top_k=50, use_history=False)
    history_predictions = enriched.predict(queries, top_k=50, use_history=True)
    location_predictions = enriched.predict(
        queries,
        top_k=50,
        use_history=True,
        use_location=True,
    )
    print(f"Enriched text TF-IDF Recall@50: {recall_at_k(text_predictions, targets):.5f}")
    print(f"Enriched text + history Recall@50: {recall_at_k(history_predictions, targets):.5f}")
    print(f"Location reranking Recall@50: {recall_at_k(location_predictions, targets):.5f}")


if __name__ == "__main__":
    main()
