from pathlib import Path

import pandas as pd


def load_data(data_dir: str | Path):
    data_dir = Path(data_dir)
    train = pd.read_parquet(
        data_dir / "train.parquet",
        columns=["item_id", "search_query", "search_category"],
    )
    queries = pd.read_parquet(
        data_dir / "benchmark_queries.parquet",
        columns=["query_id", "search_query", "search_infm_params_text", "search_category"],
    )
    items = pd.read_parquet(
        data_dir / "benchmark_items.parquet",
        columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"],
    )
    return train, queries, items


def save_answer(query_ids, predictions, output_path: str | Path):
    answer = pd.DataFrame(
        {
            "query_id": pd.Series(query_ids, dtype="string"),
            "answer": [" ".join(map(str, item_ids)) for item_ids in predictions],
        }
    )
    answer.to_csv(output_path, index=False, encoding="utf-8")
