import argparse
from pathlib import Path

from src.io import load_benchmark_data, save_answer
from src.retriever import TfidfRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Генерация кандидатов для benchmark-запросов")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--variant", choices=["baseline", "text"], default="text")
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    queries, items = load_benchmark_data(args.data_dir)

    enriched_text = args.variant == "text"
    retriever = TfidfRetriever(
        enriched_text=enriched_text,
        batch_size=args.batch_size,
    ).fit(items)
    predictions = retriever.predict(queries, top_k=50)
    save_answer(queries["query_id"], predictions, args.output)
    print(f"Сохранено {len(predictions)} ответов в {args.output}")


if __name__ == "__main__":
    main()
