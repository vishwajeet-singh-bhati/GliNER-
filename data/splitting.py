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
    return int(digest[:8], 16) / 0x100000000  # 2^32: guarantees bucket is in [0, 1)


def assign_split(record_id: str, train_frac: float = 0.7, val_frac: float = 0.15) -> str:
    bucket = split_bucket(record_id)
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"


def assign_split_campaign(record_id: str, val_frac_of_train: float = 0.15) -> str:
    """
    Matches the binary-BERT preprocessing script's split exactly:
    auto_anteater -> train (minus a val carve-out), manual_anteater -> test.
    record_id is "<file_path>::<key>", and file_path contains the campaign
    folder name, so this needs no extra data.

    Known limitation, carried over verbatim from that script's own
    documentation because it applies identically here: this is a split by
    collection campaign, not by app. 73 package names appear in both
    campaigns (~71.6% of test packages also appear in train), so a model can
    partly learn per-app traffic signatures rather than purely generalizing
    from content. Test-set metrics under this split are likely somewhat
    optimistic relative to a genuinely unseen-app scenario for that reason --
    true for the reference BERT model and equally true for anything
    evaluated on this same split.

    Use --split-strategy hash instead for an app-content-agnostic random
    split (still not app-disjoint, but at least not campaign-correlated).
    """
    if "manual_anteater" in record_id:
        return "test"
    if "auto_anteater" in record_id:
        # carve a val set out of train using the hash split, so campaign
        # mode still gets a validation set without touching manual_anteater
        return "val" if split_bucket(record_id) < val_frac_of_train else "train"
    raise ValueError(
        f"record_id '{record_id}' doesn't contain a known campaign folder "
        f"(auto_anteater/manual_anteater) -- can't assign a campaign split."
    )
