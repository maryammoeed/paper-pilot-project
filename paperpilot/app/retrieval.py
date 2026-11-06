"""A small, dependency-free BM25 index over paper chunks.

Agents query it to ground their claims in the paper's own text. Swap in a vector store
(e.g. embeddings + Qdrant/Chroma) by implementing the same `search` signature.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of and or to in on for with by is are was were be been this that we our as at from it its".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.docs = [tokenize(c.text + " " + c.section) for c in chunks]
        self.k1, self.b = k1, b
        self.avgdl = (sum(map(len, self.docs)) / len(self.docs)) if self.docs else 0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((self.chunks[i], s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
