import argparse
import re
from pathlib import Path

import pandas as pd


ITEM_ID_RE = re.compile(r"[0-9a-f]{16}")


def parse_args():
    parser = argparse.ArgumentParser(description="Check submission file format")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def check_file(path: Path, query_ids: set[str], item_ids: set[str]):
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert answer.columns.tolist() == ["query_id", "answer"], "unexpected columns"
    assert len(answer) == len(query_ids), "wrong number of rows"
    assert answer["query_id"].is_unique, "query_id values are not unique"
    assert set(answer["query_id"]) == query_ids, "query_id values do not match benchmark"

    for value in answer["answer"]:
        ids = value.split()
        assert len(ids) <= 50, "more than 50 item_id values"
        assert len(ids) == len(set(ids)), "duplicate item_id values"
        assert all(ITEM_ID_RE.fullmatch(item_id) for item_id in ids), "invalid item_id format"
        assert set(ids).issubset(item_ids), "unknown item_id"

    print(f"OK: {path} ({len(answer)} rows)")


def main():
    args = parse_args()
    query_ids = set(
        pd.read_parquet(args.data_dir / "benchmark_queries.parquet", columns=["query_id"])[
            "query_id"
        ].astype(str)
    )
    item_ids = set(
        pd.read_parquet(args.data_dir / "benchmark_items.parquet", columns=["item_id"])[
            "item_id"
        ].astype(str)
    )

    for path in args.paths:
        check_file(path, query_ids, item_ids)


if __name__ == "__main__":
    main()
