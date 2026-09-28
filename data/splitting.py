"""
Single source of truth for train/val/test assignment, shared by
convert_to_gliner_format.py (eval format) and prepare_finetune_data.py
(fine-tuning format) so both pipelines split the same underlying records the
same way -- otherwise "test_gliner.jsonl" for zero-shot eval and
"test_finetune.jsonl" for the fine-tuned model's eval could silently
correspond to different records, or worse, the eval-format test set could
include records the fine-tuned model was trained on.
"""
import hashlib


def split_bucket(record_id: str) -> float:
    """Deterministic pseudo-random value in [0, 1) from a stable hash --
    stable across runs and across processes, unlike Python's randomized
    str hash."""
    digest = hashlib.md5(record_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_split(record_id: str, train_frac: float = 0.7, val_frac: float = 0.15) -> str:
    bucket = split_bucket(record_id)
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"
