from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import build_item_text, build_query_text, make_history_key


class TfidfRetriever:
    def __init__(
        self,
        enriched_text: bool = True,
        max_features: int = 180_000,
        batch_size: int = 32,
    ):
        self.enriched_text = enriched_text
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.995,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )

    def fit(self, items: pd.DataFrame, train: pd.DataFrame | None = None):
        self.items = items.drop_duplicates("item_id").reset_index(drop=True).copy()
        self.item_ids = self.items["item_id"].astype(str).to_numpy()
        self.item_id_set = set(self.item_ids)
        self.item_matrix = self.vectorizer.fit_transform(
            build_item_text(self.items, enriched=self.enriched_text)
        )
        self._fit_history(train)
        return self

    def predict(
        self,
        queries: pd.DataFrame,
        top_k: int = 50,
        use_history: bool = True,
    ) -> list[list[str]]:
        query_matrix = self.vectorizer.transform(
            build_query_text(queries, enriched=self.enriched_text)
        )
        predictions = []

        for start in range(0, len(queries), self.batch_size):
            end = min(start + self.batch_size, len(queries))
            scores = (query_matrix[start:end] @ self.item_matrix.T).toarray()

            count = min(top_k, len(self.item_ids))
            top_indices = np.argpartition(scores, -count, axis=1)[:, -count:]
            top_scores = np.take_along_axis(scores, top_indices, axis=1)
            order = np.argsort(top_scores, axis=1)[:, ::-1]
            top_indices = np.take_along_axis(top_indices, order, axis=1)

            for local_index, (_, query) in enumerate(queries.iloc[start:end].iterrows()):
                history = self.history.get(make_history_key(query), []) if use_history else []
                lexical = self.item_ids[top_indices[local_index]].tolist()
                predictions.append(self._merge_unique(history, lexical, top_k))

        return predictions

    def _fit_history(self, train: pd.DataFrame | None):
        self.history = defaultdict(list)
        if train is None or train.empty:
            return

        counts = defaultdict(Counter)
        matched = train[train["item_id"].astype(str).isin(self.item_id_set)]
        for _, row in matched.iterrows():
            counts[make_history_key(row)][str(row["item_id"])] += 1

        for key, item_counts in counts.items():
            self.history[key] = [item_id for item_id, _ in item_counts.most_common(10)]

    @staticmethod
    def _merge_unique(first: list[str], second: list[str], top_k: int) -> list[str]:
        result = []
        seen = set()
        for item_id in first + second:
            if item_id not in seen:
                result.append(item_id)
                seen.add(item_id)
            if len(result) == top_k:
                break
        return result
