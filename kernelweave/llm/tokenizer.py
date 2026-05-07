from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import re
from collections import Counter

from .config import TokenizerConfig

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")
BYTE_TOKEN_RE = re.compile(r"^<0x([0-9a-fA-F]{2})>$")
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")


@dataclass
class TokenizerReport:
    documents: int
    observed_tokens: int
    known_tokens: int
    unknown_tokens: int
    unknown_rate: float
    vocab_size: int
    merges: int
    avg_tokens_per_doc: float


@dataclass
class TokenizerState:
    config: dict[str, Any]
    vocab: list[str]
    frequencies: dict[str, int]
    merges: list[tuple[str, str, str]]


class SimpleTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.config.validate()
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: list[str] = []
        self.frequencies: Counter[str] = Counter()
        self.merges: dict[tuple[str, str], str] = {}
        self._bootstrap_vocab()

    def _bootstrap_vocab(self) -> None:
        for token in self.config.special_tokens:
            self._add_token(token)
        if self.config.byte_fallback:
            for byte in range(256):
                self._add_token(self._byte_token(byte))

    def _byte_token(self, byte: int) -> str:
        return f"<0x{byte:02x}>"

    def _add_token(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        index = len(self.inverse_vocab)
        self.vocab[token] = index
        self.inverse_vocab.append(token)
        return index

    def _basic_tokenize(self, text: str) -> list[str]:
        source = text.lower() if self.config.lowercase else text
        return TOKEN_RE.findall(source)

    def _pair_counts(self, tokens: list[str]) -> Counter[tuple[str, str]]:
        return Counter(zip(tokens, tokens[1:]))

    def fit(self, corpus: Iterable[str]) -> TokenizerReport:
        documents = 0
        observed = 0
        known = 0
        unknown = 0
        merges = 0
        counts: Counter[str] = Counter()
        tokenized_docs: list[list[str]] = []

        for document in corpus:
            documents += 1
            tokens = self._basic_tokenize(document)
            tokenized_docs.append(tokens)
            counts.update(tokens)
            observed += len(tokens)

        capacity = max(0, self.config.vocab_size - len(self.inverse_vocab))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for token, freq in ranked:
            if capacity <= 0:
                break
            if token not in self.vocab and freq >= self.config.min_pair_frequency:
                self._add_token(token)
                self.frequencies[token] = freq
                capacity -= 1
                merges += 1

        pair_counts: Counter[tuple[str, str]] = Counter()
        for tokens in tokenized_docs:
            pair_counts.update(self._pair_counts(tokens))
        merge_rounds = 0
        while merge_rounds < self.config.max_merge_rounds and capacity > 0 and pair_counts:
            (left, right), freq = pair_counts.most_common(1)[0]
            if freq < self.config.min_pair_frequency:
                break
            merged = f"{left}::{right}"
            if merged in self.vocab:
                del pair_counts[(left, right)]
                merge_rounds += 1
                continue
            self._add_token(merged)
            self.merges[(left, right)] = merged
            capacity -= 1
            merges += 1
            merge_rounds += 1
            break

        for token, freq in counts.items():
            if token in self.vocab:
                known += freq
            else:
                unknown += freq

        total_vocab = len(self.inverse_vocab)
        return TokenizerReport(
            documents=documents,
            observed_tokens=observed,
            known_tokens=known,
            unknown_tokens=unknown,
            unknown_rate=(unknown / observed) if observed else 0.0,
            vocab_size=total_vocab,
            merges=merges,
            avg_tokens_per_doc=(observed / documents) if documents else 0.0,
        )

    def token_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("<unk>", 0))

    def token_for_id(self, token_id: int) -> str:
        if 0 <= token_id < len(self.inverse_vocab):
            return self.inverse_vocab[token_id]
        return "<unk>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        source = text.lower() if self.config.lowercase else text
        pieces = self._basic_tokenize(source)
        encoded: list[int] = []
        if add_special_tokens:
            encoded.append(self.token_id("<bos>"))
        for piece in pieces:
            if piece in self.vocab:
                encoded.append(self.vocab[piece])
                continue
            if self.config.byte_fallback:
                raw = piece.encode("utf-8", errors="replace")
                encoded.extend(self.vocab[self._byte_token(byte)] for byte in raw)
            else:
                encoded.append(self.token_id("<unk>"))
        if add_special_tokens:
            encoded.append(self.token_id("<eos>"))
        return encoded

    def _merge_pieces(self, pieces: list[str]) -> str:
        if not pieces:
            return ""
        text = " ".join(pieces)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+([\]\}])", r"\1", text)
        text = re.sub(r"([\[\{])\s+", r"\1", text)
        text = re.sub(r"\s+'", "'", text)
        return text.strip()

    def decode(self, ids: Iterable[int]) -> str:
        pieces: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            token = self.token_for_id(int(token_id))
            if token in self.config.special_tokens:
                continue
            match = BYTE_TOKEN_RE.match(token)
            if match:
                byte_buffer.append(int(match.group(1), 16))
                continue
            flush_bytes()
            pieces.append(token)

        flush_bytes()
        return self._merge_pieces(pieces)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = TokenizerState(config=self.config.to_dict(), vocab=self.inverse_vocab, frequencies=dict(self.frequencies), merges=[(a, b, c) for (a, b), c in self.merges.items()])
        path.write_text(json.dumps(state.__dict__, indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        payload = json.loads(path.read_text())
        tokenizer = cls(TokenizerConfig(**payload["config"]))
        tokenizer.vocab.clear()
        tokenizer.inverse_vocab.clear()
        for token in payload["vocab"]:
            tokenizer._add_token(token)
        tokenizer.frequencies = Counter(payload.get("frequencies", {}))
        tokenizer.merges = {}
        for left, right, merged in payload.get("merges", []):
            tokenizer.merges[(left, right)] = merged
        return tokenizer
