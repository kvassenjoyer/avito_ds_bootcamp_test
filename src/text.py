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


def build_query_text(frame: pd.DataFrame) -> list[str]:
    return frame["search_query"].map(normalize_text).tolist()


def build_item_text(frame: pd.DataFrame) -> list[str]:
    return frame["item_title_raw"].map(normalize_text).tolist()
