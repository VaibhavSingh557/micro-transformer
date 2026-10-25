"""Word-level Byte-Pair Encoding tokenizer, implemented from scratch."""
from __future__ import annotations

from collections import Counter

END = "</w>"
UNK = "<unk>"


class BPETokenizer:
    UNK_ID = 0

    def __init__(self, num_merges: int = 100) -> None:
        self.num_merges = num_merges
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {UNK: self.UNK_ID}
        self.inv_vocab: dict[int, str] = {self.UNK_ID: UNK}

    @staticmethod
    def _words(text: str) -> list[str]:
        return text.lower().split()

    @staticmethod
    def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        a, b = pair
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def train(self, corpus: str) -> None:
        words = [list(w) + [END] for w in self._words(corpus)]
        if not words:
            raise ValueError("Cannot train the tokenizer on an empty corpus.")

        self.merges = []
        for _ in range(self.num_merges):
            pairs = Counter()
            for w in words:
                pairs.update(zip(w, w[1:]))
            if not pairs:
                break
            # Deterministic tie-breaking keeps training reproducible.
            best = max(pairs, key=lambda p: (pairs[p], p))
            self.merges.append(best)
            words = [self._apply_merge(w, best) for w in words]

        symbols = sorted({s for w in words for s in w})
        self.vocab = {UNK: self.UNK_ID}
        for idx, sym in enumerate(symbols, start=1):
            self.vocab[sym] = idx
        self.inv_vocab = {idx: sym for sym, idx in self.vocab.items()}

    def encode(self, text: str) -> tuple[list[int], list[str]]:
        ids: list[int] = []
        pieces: list[str] = []
        for word in self._words(text):
            symbols = list(word) + [END]
            for pair in self.merges:
                symbols = self._apply_merge(symbols, pair)
            for sym in symbols:
                ids.append(self.vocab.get(sym, self.UNK_ID))
                pieces.append(sym)
        return ids, pieces

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.inv_vocab.get(i, UNK) for i in ids)
        return " ".join(text.replace(END, " ").split())

    def piece(self, idx: int) -> str:
        """Human-readable subword for display (end-of-word marker removed)."""
        return self.inv_vocab.get(idx, UNK).replace(END, "")
