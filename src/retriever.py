import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import build_item_text, build_query_text


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

    def fit(self, items: pd.DataFrame):
        self.items = items.drop_duplicates("item_id").reset_index(drop=True).copy()
        self.item_ids = self.items["item_id"].astype(str).to_numpy()
        self.item_matrix = self.vectorizer.fit_transform(
            build_item_text(self.items, enriched=self.enriched_text)
        )
        return self

    def predict(self, queries: pd.DataFrame, top_k: int = 50) -> list[list[str]]:
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

            for indices in top_indices:
                predictions.append(self.item_ids[indices].tolist())

        return predictions
