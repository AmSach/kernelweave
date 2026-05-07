from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "build",
    "by",
    "can",
    "create",
    "do",
    "for",
    "from",
    "get",
    "how",
    "in",
    "into",
    "is",
    "it",
    "make",
    "of",
    "on",
    "or",
    "prompt",
    "produce",
    "show",
    "solve",
    "task",
    "the",
    "to",
    "two",
    "use",
    "what",
    "write",
}

CANONICAL_ALIASES = {
    "artifact": "artifact",
    "artifacts": "artifact",
    "file": "artifact",
    "files": "artifact",
    "document": "artifact",
    "documents": "artifact",
    "doc": "artifact",
    "docs": "artifact",
    "compare": "compare",
    "comparison": "compare",
    "contrast": "compare",
    "diff": "compare",
    "difference": "compare",
    "differences": "compare",
    "ocean": "marine",
    "sea": "marine",
    "haiku": "poem",
    "poem": "poem",
    "poetry": "poem",
    "safe": "safe",
    "safety": "safe",
    "unsafe": "unsafe",
    "destructive": "unsafe",
    "dangerous": "unsafe",
    "shell": "command",
    "terminal": "command",
    "command": "command",
    "commands": "command",
    "script": "command",
    "scripts": "command",
    "tool": "tool",
    "tools": "tool",
    "agent": "agent",
    "agents": "agent",
    "kernel": "kernel",
    "kernels": "kernel",
    "store": "memory",
    "memory": "memory",
    "retrieve": "retrieve",
    "retrieval": "retrieve",
    "fallback": "fallback",
    "generate": "generate",
    "plan": "plan",
    "verify": "verify",
    "validation": "verify",
    "evidence": "evidence",
    "proof": "evidence",
    "trace": "trace",
    "traces": "trace",
    "promote": "promote",
    "compile": "compile",
    "compiler": "compile",
    "route": "route",
    "routing": "route",
    "execute": "execute",
    "execution": "execute",
    "run": "execute",
    "act": "execute",
    "actuator": "execute",
    "observe": "observe",
    "observation": "observe",
    "simulate": "simulate",
    "real": "real",
    "llm": "llm",
    "model": "model",
    "models": "model",
    "confidence": "confidence",
    "calibrate": "calibrate",
    "empirical": "empirical",
    "auto": "auto",
    "automatic": "auto",
}

INTENT_ALIASES = {
    "compare": "compare",
    "summarize": "summarize",
    "summary": "summarize",
    "extract": "extract",
    "transform": "transform",
    "convert": "transform",
    "plan": "plan",
    "route": "route",
    "select": "route",
    "choose": "route",
    "classify": "classify",
    "debug": "debug",
    "repair": "debug",
    "verify": "verify",
    "validate": "verify",
    "search": "search",
    "retrieve": "search",
    "generate": "generate",
    "draft": "generate",
    "write": "generate",
    "execute": "execute",
    "run": "execute",
    "compile": "compile",
    "learn": "learn",
    "promote": "promote",
}

OBJECT_ALIASES = {
    "artifact": "artifact",
    "memory": "memory",
    "kernel": "kernel",
    "trace": "trace",
    "command": "command",
    "code": "code",
    "data": "data",
    "model": "model",
    "paper": "paper",
    "prompt": "prompt",
    "report": "report",
    "document": "artifact",
    "doc": "artifact",
    "file": "artifact",
    "files": "artifact",
    "task": "task",
    "analysis": "analysis",
    "summary": "summary",
    "comparison": "comparison",
    "diff": "comparison",
    "extract": "extract",
    "plan": "plan",
}

ACTION_ALIASES = {
    "compare": "compare",
    "summarize": "summarize",
    "extract": "extract",
    "transform": "transform",
    "route": "route",
    "plan": "plan",
    "execute": "execute",
    "verify": "verify",
    "compile": "compile",
    "generate": "generate",
    "debug": "debug",
    "search": "search",
    "retrieve": "search",
    "promote": "promote",
}

STYLE_ALIASES = {
    "json": "json",
    "markdown": "markdown",
    "table": "table",
    "bullets": "list",
    "bullet": "list",
    "list": "list",
    "csv": "csv",
    "yaml": "yaml",
}

NEGATIVE_TERMS = {
    "ambiguous",
    "contradiction",
    "contradictory",
    "dangerous",
    "destructive",
    "failure",
    "halt",
    "invalid",
    "reject",
    "rollback",
    "stale",
    "unsafe",
    "unrelated",
    "violate",
}

INTENTS = (
    "compare",
    "summarize",
    "extract",
    "transform",
    "plan",
    "route",
    "classify",
    "debug",
    "verify",
    "search",
    "generate",
    "execute",
    "compile",
    "learn",
    "promote",
)

OBJECTS = (
    "artifact",
    "memory",
    "kernel",
    "trace",
    "command",
    "code",
    "data",
    "model",
    "paper",
    "prompt",
    "report",
    "task",
    "analysis",
    "summary",
    "comparison",
    "extract",
    "plan",
)

ACTIONS = (
    "compare",
    "summarize",
    "extract",
    "transform",
    "route",
    "plan",
    "execute",
    "verify",
    "compile",
    "generate",
    "debug",
    "search",
    "promote",
)

STYLES = ("json", "markdown", "table", "list", "csv", "yaml")

QUALITIES = (
    "safe",
    "unsafe",
    "empirical",
    "calibrated",
    "auto",
    "real",
    "fallback",
    "llm",
    "tool",
    "evidence",
    "confidence",
)


@dataclass(frozen=True)
class SemanticProfile:
    intent: str
    objects: tuple[str, ...]
    actions: tuple[str, ...]
    styles: tuple[str, ...]
    qualities: tuple[str, ...]
    terms: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def canonical_token(token: str) -> str:
    return CANONICAL_ALIASES.get(token.lower(), token.lower())


def canonical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(text):
        canonical = canonical_token(token)
        if canonical and canonical not in STOPWORDS:
            tokens.append(canonical)
    return tokens


def signature_terms(text: str, limit: int = 48) -> list[str]:
    tokens = canonical_tokens(text)
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            terms.append(token)
            seen.add(token)
        if len(terms) >= limit:
            return terms[:limit]
    for left, right in zip(tokens, tokens[1:]):
        bigram = f"{left}_{right}"
        if bigram not in seen:
            terms.append(bigram)
            seen.add(bigram)
        if len(terms) >= limit:
            return terms[:limit]
    return terms[:limit]


def _pick_first(tokens: list[str], aliases: dict[str, str], default: str) -> str:
    for token in tokens:
        if token in aliases:
            return aliases[token]
    return default


def semantic_profile(text: str) -> SemanticProfile:
    tokens = canonical_tokens(text)
    objects = tuple(dict.fromkeys(OBJECT_ALIASES[token] for token in tokens if token in OBJECT_ALIASES))
    actions = tuple(dict.fromkeys(ACTION_ALIASES[token] for token in tokens if token in ACTION_ALIASES))
    styles = tuple(dict.fromkeys(STYLE_ALIASES[token] for token in tokens if token in STYLE_ALIASES))
    qualities = tuple(dict.fromkeys(token for token in tokens if token in QUALITIES or token in NEGATIVE_TERMS))
    intent = _pick_first(tokens, INTENT_ALIASES, "general")
    terms = tuple(signature_terms(text))
    return SemanticProfile(intent=intent, objects=objects, actions=actions, styles=styles, qualities=qualities, terms=terms)


def semantic_embedding(text: str) -> list[float]:
    profile = semantic_profile(text)
    vector = [0.0] * (len(INTENTS) + len(OBJECTS) + len(ACTIONS) + len(STYLES) + len(QUALITIES) + 4)
    offset = 0

    if profile.intent in INTENTS:
        vector[offset + INTENTS.index(profile.intent)] = 1.0
    offset += len(INTENTS)

    for value in profile.objects:
        if value in OBJECTS:
            vector[offset + OBJECTS.index(value)] = 1.0
    offset += len(OBJECTS)

    for value in profile.actions:
        if value in ACTIONS:
            vector[offset + ACTIONS.index(value)] = 1.0
    offset += len(ACTIONS)

    for value in profile.styles:
        if value in STYLES:
            vector[offset + STYLES.index(value)] = 1.0
    offset += len(STYLES)

    for value in profile.qualities:
        if value in QUALITIES:
            vector[offset + QUALITIES.index(value)] = 1.0
    offset += len(QUALITIES)

    if profile.intent != "general":
        vector[offset] = 1.0
    vector[offset + 1] = 1.0 if "unsafe" in profile.qualities else 0.0
    vector[offset + 2] = 1.0 if "safe" in profile.qualities else 0.0
    vector[offset + 3] = min(1.0, len(profile.terms) / 20.0)
    return vector


def _vector_cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must be the same length")
    numerator = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def token_set(text: str) -> set[str]:
    return set(signature_terms(text))


def jaccard_similarity(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine_similarity(left: str, right: str) -> float:
    left_tokens = Counter(signature_terms(left))
    right_tokens = Counter(signature_terms(right))
    if not left_tokens or not right_tokens:
        return 0.0
    shared = set(left_tokens) & set(right_tokens)
    numerator = sum(left_tokens[token] * right_tokens[token] for token in shared)
    left_norm = math.sqrt(sum(count * count for count in left_tokens.values()))
    right_norm = math.sqrt(sum(count * count for count in right_tokens.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def semantic_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_profile = semantic_profile(left)
    right_profile = semantic_profile(right)
    vector_score = _vector_cosine(semantic_embedding(left), semantic_embedding(right))
    intent_bonus = 1.0 if left_profile.intent == right_profile.intent and left_profile.intent != "general" else 0.0
    object_score = jaccard_similarity(" ".join(left_profile.objects), " ".join(right_profile.objects))
    action_score = jaccard_similarity(" ".join(left_profile.actions), " ".join(right_profile.actions))
    quality_score = jaccard_similarity(" ".join(left_profile.qualities), " ".join(right_profile.qualities))
    phrase_score = SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()
    return clamp(
        0.42 * vector_score
        + 0.18 * intent_bonus
        + 0.14 * object_score
        + 0.12 * action_score
        + 0.06 * quality_score
        + 0.08 * phrase_score
    )


def coverage(required: Iterable[str], haystack: str) -> float:
    required = [item for item in required if item.strip()]
    if not required:
        return 1.0
    hay_profile = semantic_profile(haystack)
    hay_terms = set(signature_terms(haystack))
    hits = 0
    for item in required:
        item_profile = semantic_profile(item)
        item_terms = set(signature_terms(item))
        semantic_hit = False
        if item_profile.intent != "general" and item_profile.intent == hay_profile.intent:
            semantic_hit = True
        elif item_profile.objects and set(item_profile.objects) & set(hay_profile.objects):
            semantic_hit = True
        elif item_profile.actions and set(item_profile.actions) & set(hay_profile.actions):
            semantic_hit = True
        elif item_terms and (item_terms <= hay_terms or any(term in hay_terms for term in item_terms)):
            semantic_hit = True
        if semantic_hit:
            hits += 1
    return hits / len(required)


def conflict_terms(text: str) -> set[str]:
    profile = semantic_profile(text)
    terms = set(profile.terms)
    terms.update(profile.qualities)
    return terms & NEGATIVE_TERMS


def infer_task_family(text: str) -> str:
    profile = semantic_profile(text)
    if profile.intent == "compare":
        return "artifact comparison" if "artifact" in profile.objects else "comparison"
    if profile.intent == "summarize":
        return "structured summarization"
    if profile.intent == "extract":
        return "structured extraction"
    if profile.intent == "execute":
        return "command execution"
    if profile.intent == "compile":
        return "trace compilation"
    if profile.intent == "plan":
        return "task planning"
    if profile.intent == "debug":
        return "debugging"
    if profile.intent == "search":
        return "retrieval"
    if profile.intent == "generate" and "command" in profile.objects:
        return "safe command synthesis"
    if profile.intent == "generate" and "paper" in profile.objects:
        return "paper drafting"
    return profile.intent.replace("_", " ") if profile.intent != "general" else "general task"


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
