"""
GLiNER's training format needs token-level indices, not character offsets:

    {"tokenized_text": ["Barack", "Obama", "was", "born"],
     "ner": [[0, 1, "person"]]}   # start/end are TOKEN indices, end inclusive

This tokenizer regex is copied verbatim from GLiNER's own WhitespaceTokenSplitter
(gliner/data_processing/tokenizer.py, verified against gliner 0.2.29), so
training-time tokenization matches inference-time tokenization exactly. Because
every token produced here is either one `\\w+(?:[-_]\\w+)*` run or a single
non-space character, ' '.join(tokens) re-splits into the SAME tokens.
"""
import re
from typing import List, Optional, Tuple

_TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")

Token = Tuple[str, int, int]


def tokenize_with_offsets(text: str) -> List[Token]:
    """Returns list of (token_str, char_start, char_end)."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]


def char_span_to_token_span(
    tokens: List[Token], char_start: int, char_end: int, strict: bool = True
) -> Optional[Tuple[int, int]]:
    """
    Maps a character span [char_start, char_end) to an inclusive
    [start_token_idx, end_token_idx] span.

    strict=True (default): the span must START exactly at a token start and END
    exactly at a token end; otherwise None is returned, so callers can drop or log
    the span instead of silently labelling a larger token (BUG-002: the previous
    version accepted any token that merely *contained* the boundary character, so
    e.g. the span "12ab34" inside the token "xx12ab34cd" was labelled as the whole
    token, which happened for 7.5% of all spans after masking).

    strict=False: legacy behaviour (enclosing tokens), kept for diagnostics only.
    """
    start_idx = None
    end_idx = None
    for i, (_, tok_start, tok_end) in enumerate(tokens):
        if start_idx is None and tok_start <= char_start < tok_end:
            if strict and tok_start != char_start:
                return None
            start_idx = i
        if tok_start <= char_end - 1 < tok_end:  # [start,end) span: end token contains char_end-1
            if strict and tok_end != char_end:
                return None
            end_idx = i
    if start_idx is None or end_idx is None or end_idx < start_idx:
        return None
    return start_idx, end_idx
