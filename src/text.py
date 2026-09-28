import re

import pandas as pd


SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w]+")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def build_query_text(frame: pd.DataFrame, enriched: bool = True) -> list[str]:
    query = frame["search_query"].map(normalize_text)
    if not enriched:
        return query.tolist()
    filters = frame["search_infm_params_text"].map(normalize_text)
    return (query + " " + query + " " + filters).str.strip().tolist()


def build_item_text(frame: pd.DataFrame, enriched: bool = True) -> list[str]:
    title = frame["item_title_raw"].map(normalize_text)
    if not enriched:
        return title.tolist()
    params = frame["item_infm_params_text"].map(normalize_text).str[:300]
    description = frame["item_description_raw"].map(normalize_text).str[:500]
    return (title + " " + title + " " + params + " " + description).str.strip().tolist()


def make_exact_query_key(row: pd.Series) -> tuple:
    columns = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    return tuple(normalize_text(row[column]) for column in columns)
