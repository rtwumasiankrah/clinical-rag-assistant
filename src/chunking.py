"""Token-aware recursive text chunking.

Splits long documents into overlapping chunks measured in tokens (not
characters) so every chunk fits comfortably inside the embedding model's
input window while preserving context across chunk boundaries.
"""
from dataclasses import dataclass

import tiktoken

_SEPARATORS = ["\n\n", "\n", " ", ""]


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int


def _split_by_separators(text: str, separators: list) -> list:
    """Recursively split text on the first separator that appears in it."""
    if not separators:
        return [text]
    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        return list(text)
    if sep not in text:
        return _split_by_separators(text, rest)
    parts, splits = [], []
    for piece in text.split(sep):
        sub = _split_by_separators(piece, rest)
        splits.extend(sub)
    # re-attach the separator to every piece except the last
    for i, piece in enumerate(splits):
        parts.append(piece if i == len(splits) - 1 else piece + sep)
    return [p for p in parts if p]


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    encoding_name: str = "cl100k_base",
) -> list:
    """Split *text* into token-bounded chunks with overlap.

    Mirrors the behaviour of the recursive character splitter used in the
    original coursework: chunks are at most ``chunk_size`` tokens long and
    consecutive chunks share ``chunk_overlap`` tokens of context.
    """
    if not text or not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    encoding = tiktoken.get_encoding(encoding_name)
    token_len = lambda s: len(encoding.encode(s))  # noqa: E731

    raw_splits = _split_by_separators(text, _SEPARATORS)

    chunks, current, current_len = [], [], 0
    for piece in raw_splits:
        piece_len = token_len(piece)
        if piece_len > chunk_size:
            # Fall back to hard token windows for pathological long tokens.
            tokens = encoding.encode(piece)
            for i in range(0, len(tokens), chunk_size - chunk_overlap):
                window = tokens[i : i + chunk_size]
                chunks.append(encoding.decode(window))
            continue
        if current and current_len + piece_len > chunk_size:
            chunks.append("".join(current))
            # carry overlap into the next chunk
            overlap, overlap_len = [], 0
            for prev in reversed(current):
                prev_len = token_len(prev)
                if overlap_len + prev_len > chunk_overlap:
                    break
                overlap.insert(0, prev)
                overlap_len += prev_len
            current, current_len = overlap, overlap_len
        current.append(piece)
        current_len += piece_len
    if current:
        chunks.append("".join(current))
    return [c for c in chunks if c.strip()]


def chunk_pages(pages: list, chunk_size: int = 512, chunk_overlap: int = 20) -> list:
    """Chunk a list of ``(page_number, page_text)`` pairs, keeping provenance."""
    chunks = []
    for page_number, page_text in pages:
        for i, text in enumerate(split_text(text=page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            chunks.append(Chunk(text=text, page=page_number, chunk_index=i))
    return chunks
