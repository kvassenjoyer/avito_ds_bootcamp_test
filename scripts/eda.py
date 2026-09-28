import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DATA_DIR = Path("data")
REPORT_DIR = Path("reports")

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

HISTORY_QUERY_RE = re.compile(r"[^0-9a-zа-я]+")


def normalize_history_query(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().replace("ё", "е")
    return HISTORY_QUERY_RE.sub(" ", value).strip()


def parquet_info(path):
    parquet_file = pq.ParquetFile(path)
    null_counts = {}
    for index, column in enumerate(parquet_file.schema.names):
        statistics = parquet_file.metadata.row_group(0).column(index).statistics
        null_counts[column] = (
            statistics.null_count
            if statistics is not None and statistics.has_null_count
            else None
        )
    return {
        "rows": parquet_file.metadata.num_rows,
        "columns": {
            field.name: str(field.type) for field in parquet_file.schema_arrow
        },
        "null_counts": null_counts,
    }


def text_missing(path, columns):
    result = {}
    for column in columns:
        values = pd.read_parquet(path, columns=[column])[column].astype("string")
        result[column] = {
            "null": int(values.isna().sum()),
            "blank": int(values.fillna("").str.strip().eq("").sum()),
        }
    return result


def describe_counts(values):
    return {
        "min": int(values.min()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "p99": float(values.quantile(0.99)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def build_report():
    train_path = DATA_DIR / "train.parquet"
    query_path = DATA_DIR / "benchmark_queries.parquet"
    item_path = DATA_DIR / "benchmark_items.parquet"

    train = pd.read_parquet(
        train_path,
        columns=QUERY_COLUMNS + ["item_id", "item_category_id"],
    )
    benchmark_queries = pd.read_parquet(query_path)
    benchmark_items = pd.read_parquet(
        item_path,
        columns=["item_id", "item_category_id"],
    )

    train_group_sizes = train.groupby(QUERY_COLUMNS, dropna=False).size()
    train_positive_counts = (
        train.groupby(QUERY_COLUMNS, dropna=False)["item_id"].nunique()
    )

    train_signatures = train[QUERY_COLUMNS].drop_duplicates()
    benchmark_signatures = benchmark_queries[QUERY_COLUMNS].drop_duplicates()
    signature_overlap = benchmark_signatures.merge(train_signatures, on=QUERY_COLUMNS)

    train_queries = set(train["search_query"].dropna())
    benchmark_query_texts = set(benchmark_queries["search_query"].dropna())
    train_items = set(train["item_id"].dropna())
    corpus_items = set(benchmark_items["item_id"].dropna())

    history_train = train[train["item_id"].isin(corpus_items)].copy()
    history_train["normalized_query"] = history_train["search_query"].map(
        normalize_history_query
    )
    history_sizes = (
        history_train.groupby(["normalized_query", "search_category"])["item_id"]
        .nunique()
        .to_dict()
    )
    benchmark_history_sizes = pd.Series(
        [
            history_sizes.get((normalize_history_query(query), category), 0)
            for query, category in zip(
                benchmark_queries["search_query"],
                benchmark_queries["search_category"],
            )
        ]
    )
    covered_history_sizes = benchmark_history_sizes[benchmark_history_sizes > 0]

    candidate_counts = (
        benchmark_items.groupby("item_category_id")["item_id"].nunique()
    )
    benchmark_candidate_counts = benchmark_queries["search_category"].map(candidate_counts)
    benchmark_candidate_counts = benchmark_candidate_counts.where(
        benchmark_queries["search_category"].ne(0),
        len(benchmark_items),
    )

    report = {
        "files": {
            "train": parquet_info(train_path),
            "benchmark_queries": parquet_info(query_path),
            "benchmark_items": parquet_info(item_path),
        },
        "text_missing": {
            "train": text_missing(
                train_path,
                [
                    "search_query",
                    "search_infm_params_text",
                    "item_title_raw",
                    "item_description_raw",
                    "item_infm_params_text",
                ],
            ),
            "benchmark_queries": text_missing(
                query_path,
                ["search_query", "search_infm_params_text"],
            ),
            "benchmark_items": text_missing(
                item_path,
                [
                    "item_title_raw",
                    "item_description_raw",
                    "item_infm_params_text",
                ],
            ),
        },
        "train": {
            "rows": len(train),
            "unique_query_signatures": len(train_signatures),
            "unique_query_texts": int(train["search_query"].nunique(dropna=True)),
            "unique_items": len(train_items),
            "unique_search_categories": int(train["search_category"].nunique()),
            "unique_item_categories": int(train["item_category_id"].nunique()),
            "duplicate_signature_item_pairs": int(
                train.duplicated(QUERY_COLUMNS + ["item_id"]).sum()
            ),
            "search_category_counts": {
                str(key): int(value)
                for key, value in train["search_category"].value_counts().items()
            },
            "item_category_counts": {
                str(key): int(value)
                for key, value in train["item_category_id"].value_counts().items()
            },
            "rows_per_query_signature": describe_counts(train_group_sizes),
            "unique_positives_per_query_signature": describe_counts(
                train_positive_counts
            ),
            "query_signatures_with_multiple_positives": int(
                (train_positive_counts > 1).sum()
            ),
            "item_category_matches_search_category_rows": int(
                train["item_category_id"].eq(train["search_category"]).sum()
            ),
            "item_category_matches_search_category_rate": float(
                train["item_category_id"].eq(train["search_category"]).mean()
            ),
        },
        "benchmark": {
            "query_ids_unique": bool(benchmark_queries["query_id"].is_unique),
            "item_ids_unique": bool(benchmark_items["item_id"].is_unique),
            "unique_query_signatures": len(benchmark_signatures),
            "unique_query_texts": len(benchmark_query_texts),
            "unique_search_categories": int(
                benchmark_queries["search_category"].nunique()
            ),
            "unique_item_categories": int(
                benchmark_items["item_category_id"].nunique()
            ),
            "all_category_queries": int(
                benchmark_queries["search_category"].eq(0).sum()
            ),
            "search_category_counts": {
                str(key): int(value)
                for key, value in benchmark_queries["search_category"]
                .value_counts()
                .items()
            },
            "largest_item_category_counts": {
                str(key): int(value)
                for key, value in benchmark_items["item_category_id"]
                .value_counts()
                .head(10)
                .items()
            },
            "queries_without_candidates_after_category_filter": int(
                benchmark_candidate_counts.isna().sum()
            ),
            "category_filtered_candidate_counts": describe_counts(
                benchmark_candidate_counts.dropna()
            ),
            "category_114_corpus_rate": float(
                benchmark_items["item_category_id"].eq(114).mean()
            ),
        },
        "overlap": {
            "train_items_in_benchmark_corpus": len(train_items & corpus_items),
            "train_items_in_benchmark_corpus_rate": len(train_items & corpus_items)
            / len(train_items),
            "benchmark_items_seen_in_train": len(train_items & corpus_items),
            "benchmark_items_seen_in_train_rate": len(train_items & corpus_items)
            / len(corpus_items),
            "benchmark_exact_query_signatures_seen_in_train": len(signature_overlap),
            "benchmark_exact_query_signatures_seen_in_train_rate": len(
                signature_overlap
            )
            / len(benchmark_signatures),
            "benchmark_query_texts_seen_in_train": len(
                train_queries & benchmark_query_texts
            ),
            "benchmark_query_texts_seen_in_train_rate": len(
                train_queries & benchmark_query_texts
            )
            / len(benchmark_query_texts),
        },
        "history": {
            "key": "normalized search_query + search_category",
            "normalization": "lowercase, ё->е, non-alphanumeric characters to spaces",
            "benchmark_queries_covered": int(
                benchmark_history_sizes.gt(0).sum()
            ),
            "benchmark_queries_covered_rate": float(
                benchmark_history_sizes.gt(0).mean()
            ),
            "corpus_items_per_covered_query": describe_counts(
                covered_history_sizes
            ),
        },
    }
    return report


def write_markdown(report):
    train = report["train"]
    benchmark = report["benchmark"]
    overlap = report["overlap"]
    history = report["history"]
    missing = report["text_missing"]

    lines = [
        "# Краткий EDA",
        "",
        "## Что влияет на решение",
        "",
        f"- В train {train['rows']:,} строк, "
        f"{train['unique_query_signatures']:,} уникальных сигнатур запроса и "
        f"{train['unique_items']:,} уникальных объявлений.",
        f"- Число уникальных positives на сигнатуру: "
        f"median={train['unique_positives_per_query_signature']['median']:.0f}, "
        f"p90={train['unique_positives_per_query_signature']['p90']:.0f}, "
        f"p99={train['unique_positives_per_query_signature']['p99']:.0f}, "
        f"max={train['unique_positives_per_query_signature']['max']}.",
        f"- Повторных пар (полная сигнатура, item_id): "
        f"{train['duplicate_signature_item_pairs']:,}; при сборке ground truth их надо "
        "дедуплицировать.",
        f"- Полное совпадение search/item category: "
        f"{train['item_category_matches_search_category_rate']:.2%} строк.",
        f"- В benchmark {benchmark['all_category_queries']} запросов имеют "
        "search_category=0 (для них нужен весь corpus). Для category=114 точный "
        f"фильтр оставляет {benchmark['category_114_corpus_rate']:.2%} corpus, то есть "
        "почти не ускоряет поиск.",
        f"- В benchmark corpus присутствует "
        f"{overlap['train_items_in_benchmark_corpus_rate']:.2%} уникальных train items; "
        f"{overlap['benchmark_items_seen_in_train_rate']:.2%} corpus items встречались в train.",
        f"- Exact query signature overlap benchmark/train: "
        f"{overlap['benchmark_exact_query_signatures_seen_in_train_rate']:.2%}; "
        f"overlap только по search_query: "
        f"{overlap['benchmark_query_texts_seen_in_train_rate']:.2%}.",
        f"- History key (нормализованный search_query + category) покрывает "
        f"{history['benchmark_queries_covered']}/2452 запросов "
        f"({history['benchmark_queries_covered_rate']:.2%}); доступных corpus items "
        f"на покрытый запрос обычно "
        f"{history['corpus_items_per_covered_query']['median']:.0f}, максимум "
        f"{history['corpus_items_per_covered_query']['max']}.",
        "",
        "## Пропуски в текстах (null / blank including null)",
        "",
    ]
    for dataset, columns in missing.items():
        lines.append(f"- {dataset}: " + ", ".join(
            f"{column}={values['null']}/{values['blank']}"
            for column, values in columns.items()
        ))

    lines += [
        "",
        "## Валидация",
        "",
        "Для offline Recall@50 нужно как минимум делить данные по полной сигнатуре "
        "запроса (все search_* поля), например GroupShuffleSplit. Более строгий вариант "
        "для проверки lexical generalization — группы по search_query. Обычный random "
        "row split оставит одинаковые запросы в train и validation и завысит качество. "
        "Corpus для validation строится из уникальных объявлений всего train, "
        "а релевантность — из строк только held-out query groups.",
        "",
        "## Практический вывод",
        "",
        "Для первого воспроизводимого решения достаточно TF-IDF по объединённым текстам "
        "запроса/объявления. Category filter не включаем в основной pipeline: category=114 "
        "почти не сокращает corpus, а для category=0 всё равно нужен полный поиск. "
        "Историю item_id можно использовать "
        "как дополнительный fallback только для запросов, встречавшихся в train: "
        "точный overlap сигнатур нужно измерять отдельно от общей lexical retrieval.",
        "",
    ]
    (REPORT_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    report = build_report()
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(report)
    print("Reports saved to reports/eda_summary.json and reports/eda_summary.md")


if __name__ == "__main__":
    main()
