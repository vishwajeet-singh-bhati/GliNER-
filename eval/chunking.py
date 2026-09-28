"""
Sliding-window chunking for GLiNER's predict_entities, which truncates
anything past ~384 tokens (silently, via a UserWarning) rather than erroring.
After the header-cleaning pass in antshield_loader.py, only the top ~1% of
AntShield packets exceed a safe window size, but this makes the eval correct
for that tail rather than silently dropping their entities.

Character-based (not token-based) because predict_entities takes a raw
string and does its own tokenization internally -- we don't need to
duplicate that here, just avoid handing it more raw text than it can see.
"""
from typing import List, Dict


def chunk_text(text: str, window_chars: int = 1500, overlap_chars: int = 200) -> List[tuple]:
    """
    Yields (chunk_text, chunk_start_offset) pairs. window_chars=1500 is a
    conservative proxy for "well under 384 tokens" for this dataset's mostly-
    ASCII, punctuation-heavy packet text (~4 chars/token average observed).
    overlap_chars must be larger than the longest expected entity span so a
    PII value straddling a window boundary still appears whole in at least
    one window.
    """
    if len(text) <= window_chars:
        return [(text, 0)]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + window_chars, len(text))
        chunks.append((text[start:end], start))
        if end == len(text):
            break
        start = end - overlap_chars
    return chunks


def predict_with_chunking(model, text: str, labels, threshold: float = 0.5, **chunk_kwargs) -> List[Dict]:
    """
    Runs model.predict_entities per chunk, translates each chunk's local
    (start, end) back to the full text's coordinates, and dedupes entities
    that get predicted twice from overlapping windows (keeping the
    higher-confidence one where scores are available).
    """
    chunks = chunk_text(text, **chunk_kwargs)
    merged: Dict[tuple, Dict] = {}

    for chunk, offset in chunks:
        predicted = model.predict_entities(chunk, labels, threshold=threshold)
        for p in predicted:
            global_start = p["start"] + offset
            global_end = p["end"] + offset
            key = (global_start, global_end, p["label"])
            candidate = {
                "start": global_start,
                "end": global_end,
                "label": p["label"],
                "score": p.get("score", 0.0),
            }
            existing = merged.get(key)
            if existing is None or candidate["score"] > existing["score"]:
                merged[key] = candidate

    return list(merged.values())
