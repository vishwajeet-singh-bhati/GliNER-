"""
GLiNER's training format needs token-level indices, not character offsets:

    {"tokenized_text": ["Barack", "Obama", "was", "born"],
     "ner": [[0, 1, "person"]]}   # start/end are TOKEN indices, end inclusive

This tokenizer regex is copied verbatim from GLiNER's own predict_entities
implementation (confirmed from source), so training-time tokenization
matches inference-time tokenization exactly -- using a different tokenizer
for training data would silently misalign spans against what the model
actually sees at inference.
"""
import re
from typing import List, Optional, Tuple

_TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Returns list of (token_str, char_start, char_end)."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]


def char_span_to_token_span(
    tokens: List[Tuple[str, int, int]], char_start: int, char_end: int
) -> Optional[Tuple[int, int]]:
    """
    Maps a character span [char_start, char_end) to an inclusive
    [start_token_idx, end_token_idx] span. Returns None if the char span
    doesn't align to a clean run of tokens (e.g. it starts or ends mid-token)
    -- callers should drop or log these rather than emit a wrong span.
    """
    start_idx = None
    end_idx = None
    for i, (_, tok_start, tok_end) in enumerate(tokens):
        if start_idx is None and tok_start <= char_start < tok_end:
            start_idx = i
        if tok_start < char_end <= tok_end:
            end_idx = i
    if start_idx is None or end_idx is None or end_idx < start_idx:
        return None
    return start_idx, end_idx
