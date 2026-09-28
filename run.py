import argparse
import gc
from pathlib import Path

from src.io import load_data, save_answer
from src.retriever import TfidfRetriever


OUTPUTS = {
    "baseline": Path("submissions/answer_01_title_tfidf.csv"),
    "text": Path("submissions/answer_02_text_tfidf.csv"),
    "history": Path("submissions/answer_03_tfidf_history.csv"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidates for benchmark queries")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--variant",
        choices=["baseline", "text", "history", "all"],
        default="history",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    train, queries, items = load_data(args.data_dir)

    if args.variant == "all":
        Path("submissions").mkdir(exist_ok=True)
        baseline = TfidfRetriever(enriched_text=False, batch_size=args.batch_size).fit(items)
        baseline_predictions = baseline.predict(queries, top_k=50, use_history=False)
        save_answer(queries["query_id"], baseline_predictions, OUTPUTS["baseline"])
        del baseline, baseline_predictions
        gc.collect()

        enriched = TfidfRetriever(enriched_text=True, batch_size=args.batch_size).fit(items, train)
        del train
        gc.collect()
        text_predictions = enriched.predict(queries, top_k=50, use_history=False)
        history_predictions = enriched.predict(queries, top_k=50, use_history=True)
        save_answer(queries["query_id"], text_predictions, OUTPUTS["text"])
        save_answer(queries["query_id"], history_predictions, OUTPUTS["history"])
        save_answer(queries["query_id"], history_predictions, Path("answer.csv"))
        print("Saved three variants to submissions/ and the selected result to answer.csv")
        return

    enriched_text = args.variant != "baseline"
    use_history = args.variant == "history"
    retriever = TfidfRetriever(enriched_text=enriched_text, batch_size=args.batch_size)
    retriever.fit(items, train if use_history else None)
    del train
    gc.collect()
    predictions = retriever.predict(queries, top_k=50, use_history=use_history)

    output = args.output or OUTPUTS[args.variant]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_answer(queries["query_id"], predictions, output)
    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
