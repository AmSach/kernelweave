from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard_similarity(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine_similarity(left: str, right: str) -> float:
    left_tokens = Counter(tokenize(left))
    right_tokens = Counter(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    shared = set(left_tokens) & set(right_tokens)
    numerator = sum(left_tokens[token] * right_tokens[token] for token in shared)
    left_norm = math.sqrt(sum(count * count for count in left_tokens.values()))
    right_norm = math.sqrt(sum(count * count for count in right_tokens.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def coverage(required: Iterable[str], haystack: str) -> float:
    required = [item for item in required if item.strip()]
    if not required:
        return 1.0
    hay = token_set(haystack)
    hits = 0
    for item in required:
        item_tokens = token_set(item)
        if not item_tokens:
            continue
        if item_tokens <= hay or any(token in hay for token in item_tokens):
            hits += 1
    return hits / len(required)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
