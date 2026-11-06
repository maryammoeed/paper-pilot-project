"""Turn a PDF or text paper into section-aware chunks with stable IDs (c0, c1, ...)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

HEADING = re.compile(
    r"^\s*(?:#+\s*)?(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|related work|background|method(?:s|ology)?|approach|experiments?|"
    r"results?|evaluation|discussion|limitations?|conclusions?|references)\b.*$",
    re.IGNORECASE,
)


SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u201c\u2018])")


@dataclass
class Chunk:
    id: str
    section: str
    text: str


@dataclass
class Paper:
    title: str
    text: str
    chunks: list[Chunk]

    def chunk(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _title(lines: list[str]) -> str:
    for line in lines:
        s = line.strip().lstrip("#").strip()
        if s and not HEADING.match(s):
            return s[:200]
    return "Untitled paper"


def parse(text: str, max_words: int = 140) -> Paper:
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    title = _title(lines)
    section = "Front matter"
    chunks: list[Chunk] = []
    buf: list[str] = []

    def flush():
        # pack whole sentences into chunks of ~max_words so a quote never straddles two chunks
        piece: list[str] = []
        for sent in SENTENCE.split(" ".join(buf)):
            n = len(sent.split())
            if piece and len(" ".join(piece).split()) + n > max_words:
                chunks.append(Chunk(id=f"c{len(chunks)}", section=section, text=" ".join(piece)))
                piece = []
            if n > max_words:          # one giant "sentence" (e.g. no punctuation): split by words
                words = sent.split()
                while len(words) > max_words:
                    chunks.append(Chunk(id=f"c{len(chunks)}", section=section, text=" ".join(words[:max_words])))
                    words = words[max_words:]
                sent = " ".join(words)
            if sent.strip():
                piece.append(sent.strip())
        if piece:
            chunks.append(Chunk(id=f"c{len(chunks)}", section=section, text=" ".join(piece)))
        buf.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lstrip("#").strip() == title and not chunks and not buf:
            continue
        m = HEADING.match(stripped)
        if m and len(stripped.split()) <= 8:
            flush()
            section = stripped.lstrip("#").strip()
            if section.lower().startswith("references"):
                break  # references add noise for review
            continue
        buf.append(stripped)
    flush()
    return Paper(title=title, text=text, chunks=chunks)
