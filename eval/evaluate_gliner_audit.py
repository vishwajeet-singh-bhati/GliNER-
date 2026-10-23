#!/usr/bin/env python3
"""
Audit-grade GLiNER Evaluator (eval/evaluate_gliner_audit.py)
Read-only evaluation of base, fine-tuned, and quantized GLiNER checkpoints.
Never modifies existing datasets or checkpoints. Never logs raw PII text.

Changes vs. the first version (all of them were measured to matter, see the audit notes):
  * labels: the model is prompted with the labels it was TRAINED on (train_labels.json written by
    train/finetune_gliner.py, or --labels-file).  The old default prompted all 17 incl. one label with
    zero training/test spans, which can only add false positives.
  * chunking: GLiNER's max_len counts WORDS, not subwords.  The old chunker treated 384 as a subword
    budget minus the label-prompt overhead, which split ~half of the 300-word windows into pieces the
    model was never trained on.  Default --chunk-unit words (a 300-word window is ONE forward pass).
  * threshold: chosen on the VALIDATION file (--val-path) and then applied once, exactly, to the test
    file.  Tuning on test is never done.  A sweep table is reported for transparency.
  * error breakdown (label confusion / boundary / spurious in entity-free windows ...), per label.
  * optional batched inference (--batch-size).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "quantization"))

DEFAULT_17_LABELS: Tuple[str, ...] = (
    "advertiser id",
    "android device id",
    "city",
    "device serial number",
    "email address",
    "first name",
    "gender",
    "imei number",
    "imsi number",
    "last name",
    "location coordinates",
    "other personal information",
    "password",
    "phone number",
    "sim card iccid",
    "username",
    "zip code",
)


def get_configured_labels() -> List[str]:
    """Load the 17 canonical labels from data/pii_type_mapping.py with fallback."""
    try:
        from pii_type_mapping import all_gliner_labels
        labels = list(all_gliner_labels())
        if len(labels) != 17:
            raise ValueError(f"Expected 17 labels from pii_type_mapping, got {len(labels)}")
        return sorted(labels)
    except Exception:
        return sorted(DEFAULT_17_LABELS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ValidationSummary:
    total_lines: int = 0
    valid_examples: int = 0
    malformed_examples: int = 0
    total_raw_gold_spans: int = 0
    valid_unique_gold_spans: int = 0
    duplicate_gold_spans: int = 0
    invalid_gold_spans: int = 0
    unknown_label_spans: int = 0
    examples_with_overlapping_gold: int = 0
    overlapping_gold_span_count: int = 0
    invalid_example_indices: List[int] = None
    overlapping_example_indices: List[int] = None

    def __post_init__(self):
        if self.invalid_example_indices is None:
            self.invalid_example_indices = []
        if self.overlapping_example_indices is None:
            self.overlapping_example_indices = []


def has_overlapping_spans(spans: Sequence[Tuple[int, int, str]]) -> Tuple[bool, int]:
    """Check if any two unique intervals [s1, e1] and [s2, e2] share a token index."""
    if len(spans) <= 1:
        return False, 0
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1], x[2]))
    overlap_Ids: Set[int] = set()
    for i in range(len(sorted_spans)):
        s1, e1, _ = sorted_spans[i]
        for j in range(i + 1, len(sorted_spans)):
            s2, e2, _ = sorted_spans[j]
            if s2 <= e1:
                overlap_Ids.add(i)
                overlap_Ids.add(j)
            else:
                break
    return (len(overlap_Ids) > 0), len(overlap_Ids)


def validate_and_parse_example(
    raw_obj: Any,
    line_idx: int,
    allowed_labels: Set[str],
    summary: ValidationSummary,
) -> Optional[Tuple[List[str], List[Tuple[int, int, str]], bool]]:
    """
    Validate a single JSONL record without ever printing raw token text.
    Returns (tokens, deduplicated_valid_gold_spans, has_overlap) or None if record is malformed.
    """
    summary.total_lines += 1
    if not isinstance(raw_obj, dict):
        summary.malformed_examples += 1
        summary.invalid_example_indices.append(line_idx)
        return None

    tokens = raw_obj.get("tokenized_text")
    ner = raw_obj.get("ner")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        summary.malformed_examples += 1
        summary.invalid_example_indices.append(line_idx)
        return None
    if not isinstance(ner, list):
        summary.malformed_examples += 1
        summary.invalid_example_indices.append(line_idx)
        return None

    n_tok = len(tokens)
    seen_spans: Set[Tuple[int, int, str]] = set()
    dedup_spans: List[Tuple[int, int, str]] = []
    example_had_invalid_span = False

    for item in ner:
        summary.total_raw_gold_spans += 1
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            summary.invalid_gold_spans += 1
            example_had_invalid_span = True
            continue
        s, e, label = item
        if (
            not isinstance(s, int)
            or isinstance(s, bool)
            or not isinstance(e, int)
            or isinstance(e, bool)
            or not isinstance(label, str)
        ):
            summary.invalid_gold_spans += 1
            example_had_invalid_span = True
            continue
        if s < 0 or e < s or e >= n_tok:
            summary.invalid_gold_spans += 1
            example_had_invalid_span = True
            continue
        if label not in allowed_labels:
            summary.unknown_label_spans += 1
            summary.invalid_gold_spans += 1
            example_had_invalid_span = True
            continue

        key = (s, e, label)
        if key in seen_spans:
            summary.duplicate_gold_spans += 1
            continue
        seen_spans.add(key)
        dedup_spans.append(key)
        summary.valid_unique_gold_spans += 1

    if example_had_invalid_span:
        summary.invalid_example_indices.append(line_idx)

    overlaps, n_overlaps = has_overlapping_spans(dedup_spans)
    if overlaps:
        summary.examples_with_overlapping_gold += 1
        summary.overlapping_gold_span_count += n_overlaps
        summary.overlapping_example_indices.append(line_idx)

    summary.valid_examples += 1
    return tokens, dedup_spans, overlaps


@dataclass
class ChunkSpec:
    start_tok: int
    end_tok: int  # inclusive token index
    subword_count: int
    exceeded_single_token_budget: bool = False


class SubwordChunker:
    """
    Token-aware chunker that guarantees subword count <= model max_len
    and 100% coverage of token indices [0, len(tokens) - 1].
    """

    def __init__(
        self,
        subword_counter_fn,
        labels: Sequence[str],
        max_len: int = 384,
        safety_margin: int = 8,
        overlap_subwords: int = 48,
    ):
        self.subword_counter_fn = subword_counter_fn
        self.max_len = max_len
        self.safety_margin = safety_margin
        # GLiNER prompt structure: [CLS] <<ENT>> l1 <<ENT>> l2 ... <<SEP>> text [SEP]
        label_subwords = sum(1 + max(1, self.subword_counter_fn(l)) for l in labels) + 1
        self.prompt_overhead = label_subwords + 2  # +2 for BOS/CLS and EOS/SEP
        self.text_budget = max(16, self.max_len - self.prompt_overhead - self.safety_margin)
        self.overlap_subwords = min(overlap_subwords, max(0, self.text_budget // 3))

    def chunk_tokens(self, tokens: Sequence[str]) -> Tuple[List[ChunkSpec], List[int]]:
        if not tokens:
            return [], []
        word_subwords = [max(1, self.subword_counter_fn(w)) for w in tokens]
        total_sub = sum(word_subwords)
        n = len(tokens)

        if total_sub <= self.text_budget:
            return [
                ChunkSpec(
                    start_tok=0,
                    end_tok=n - 1,
                    subword_count=self.prompt_overhead + total_sub,
                    exceeded_single_token_budget=False,
                )
            ], word_subwords

        chunks: List[ChunkSpec] = []
        start = 0
        while start < n:
            cur_sub = 0
            end = start
            # If a single word alone exceeds the budget, isolate it in a 1-token chunk
            if word_subwords[start] > self.text_budget:
                chunks.append(
                    ChunkSpec(
                        start_tok=start,
                        end_tok=start,
                        subword_count=self.prompt_overhead + word_subwords[start],
                        exceeded_single_token_budget=True,
                    )
                )
                start += 1
                continue

            while end < n and (cur_sub + word_subwords[end]) <= self.text_budget:
                cur_sub += word_subwords[end]
                end += 1

            chunk_end_inclusive = end - 1
            chunks.append(
                ChunkSpec(
                    start_tok=start,
                    end_tok=chunk_end_inclusive,
                    subword_count=self.prompt_overhead + cur_sub,
                    exceeded_single_token_budget=False,
                )
            )
            if end >= n:
                break

            # Step back from `end` to include up to `overlap_subwords` of context,
            # while strictly ensuring start advances by at least 1 token.
            back_sub = 0
            next_start = end
            while next_start > start + 1 and (back_sub + word_subwords[next_start - 1]) <= self.overlap_subwords:
                next_start -= 1
                back_sub += word_subwords[next_start]
            start = next_start

        return chunks, word_subwords


class WordChunker:
    """Chunks by WORD count (GLiNER's max_len unit) AND caps each chunk's length in subwords.

    * <= max_len words and <= max_subwords (incl. the label prompt) -> ONE chunk (the normal case).
    * The subword cap matters: a 300-word window made of long hex/base64 tokens can be 5,000+ subwords,
      and DeBERTa's attention is quadratic in that (one such window asked for 31 GiB and crashed the
      evaluation).  Training dropped windows over 1,024 subwords, so the model never saw longer inputs;
      the default cap mirrors that.
    Same interface as SubwordChunker (chunk_tokens -> (chunks, per-word subword counts))."""

    def __init__(self, subword_counter_fn, labels, max_len: int = 384, overlap_words: int = 30,
                 max_subwords: int = 1024):
        self.subword_counter_fn = subword_counter_fn
        self.max_len = max_len
        self.overlap_words = min(overlap_words, max(0, max_len // 4))
        self.prompt_overhead = sum(1 + max(1, subword_counter_fn(l)) for l in labels) + 3
        self.sub_budget = max(16, max_subwords - self.prompt_overhead)
        self.overlap_subwords = self.sub_budget // 4
        self.text_budget = max_len        # unit: words (the subword cap is sub_budget)

    def chunk_tokens(self, tokens):
        n = len(tokens)
        if n == 0:
            return [], []
        wsub = [max(1, self.subword_counter_fn(w)) for w in tokens]
        chunks, start = [], 0
        while start < n:
            if wsub[start] > self.sub_budget:        # a single giant token: isolate it, flag it
                chunks.append(ChunkSpec(start_tok=start, end_tok=start,
                                        subword_count=self.prompt_overhead + wsub[start],
                                        exceeded_single_token_budget=True))
                start += 1
                continue
            end, cur = start, 0
            while end < n and end - start < self.max_len and cur + wsub[end] <= self.sub_budget:
                cur += wsub[end]
                end += 1
            chunks.append(ChunkSpec(start_tok=start, end_tok=end - 1, subword_count=self.prompt_overhead + cur))
            if end >= n:
                break
            nxt, back_w, back_s = end, 0, 0
            while (nxt > start + 1 and back_w < self.overlap_words
                   and back_s + wsub[nxt - 1] <= self.overlap_subwords):
                nxt -= 1
                back_w += 1
                back_s += wsub[nxt]
            start = nxt
        return chunks, wsub


def build_chunk_string_and_offsets(chunk_tokens: Sequence[str]) -> Tuple[str, List[Tuple[int, int]], Dict[int, int], Dict[int, int]]:
    """
    Join tokens with a single space (replacing any internal whitespace inside a token with '_')
    so character offsets map 1-to-1 with token boundaries.
    """
    sanitized = [
        "".join("_" if ch.isspace() else ch for ch in (t if t else "_"))
        for t in chunk_tokens
    ]
    offsets: List[Tuple[int, int]] = []
    start_map: Dict[int, int] = {}
    end_map: Dict[int, int] = {}
    cursor = 0
    for idx, tok in enumerate(sanitized):
        s = cursor
        e = s + len(tok)
        offsets.append((s, e))
        start_map[s] = idx
        end_map[e] = idx
        cursor = e + 1  # +1 for the joining space
    return " ".join(sanitized), offsets, start_map, end_map


def map_char_pred_to_tokens(
    char_start: int,
    char_end: int,
    offsets: Sequence[Tuple[int, int]],
    start_map: Dict[int, int],
    end_map: Dict[int, int],
    allow_enclosing_fallback: bool = False,
) -> Tuple[Optional[Tuple[int, int]], bool]:
    """
    Map character span [char_start, char_end) back to inclusive chunk token indices (s_tok, e_tok).
    Returns ((s_tok, e_tok), is_exact_boundary).
    """
    if char_start in start_map and char_end in end_map:
        s_tok = start_map[char_start]
        e_tok = end_map[char_end]
        if s_tok <= e_tok:
            return (s_tok, e_tok), True

    if not allow_enclosing_fallback:
        return None, False

    s_tok, e_tok = None, None
    for i, (ts, te) in enumerate(offsets):
        if ts <= char_start < te:
            s_tok = i
        if ts < char_end <= te:
            e_tok = i
    if s_tok is not None and e_tok is not None and s_tok <= e_tok:
        return (s_tok, e_tok), False
    return None, False


def deduplicate_predictions(
    preds_with_scores: Sequence[Tuple[int, int, str, float]]
) -> List[Tuple[int, int, str, float]]:
    """Deduplicate (start_tok, end_tok, label, score) across chunks, keeping max score."""
    best: Dict[Tuple[int, int, str], float] = {}
    for s, e, label, score in preds_with_scores:
        key = (s, e, label)
        if key not in best or score > best[key]:
            best[key] = score
    return [(s, e, lbl, sc) for (s, e, lbl), sc in sorted(best.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))]


def match_spans_one_to_one(
    gold_spans: Sequence[Tuple[int, int, str]],
    pred_spans: Sequence[Tuple[int, int, str]],
    labels: Sequence[str],
) -> Dict[str, Counter]:
    """
    1-to-1 exact matching by (start_tok, end_tok, label).
    Duplicate gold or predicted spans are deduplicated before matching so a single gold
    span can never be matched twice.
    """
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    stats = {l: Counter({"tp": 0, "fp": 0, "fn": 0, "support": 0, "predicted": 0}) for l in labels}

    for s, e, l in gold_set:
        if l in stats:
            stats[l]["support"] += 1
            if (s, e, l) in pred_set:
                stats[l]["tp"] += 1
            else:
                stats[l]["fn"] += 1

    for s, e, l in pred_set:
        if l in stats:
            stats[l]["predicted"] += 1
            if (s, e, l) not in gold_set:
                stats[l]["fp"] += 1

    return stats


def compute_metrics_table(
    class_totals: Dict[str, Counter],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Compute per-class TP/FP/FN/P/R/F1, micro-F1, macro-F1 (all 17), and macro-F1 (supported)."""
    per_class: Dict[str, Dict[str, Any]] = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    f1_all: List[float] = []
    f1_supported: List[float] = []

    for l in sorted(labels):
        c = class_totals.get(l, Counter())
        tp = int(c["tp"])
        fp = int(c["fp"])
        fn = int(c["fn"])
        support = int(c["support"])
        predicted = int(c["predicted"])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[l] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "predicted": predicted,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "has_support": support > 0,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        f1_all.append(f1)
        if support > 0:
            f1_supported.append(f1)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    macro_all = sum(f1_all) / len(f1_all) if f1_all else 0.0
    macro_sup = sum(f1_supported) / len(f1_supported) if f1_supported else 0.0

    return {
        "micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "support": total_tp + total_fn,
            "predicted": total_tp + total_fp,
            "precision": round(micro_p, 6),
            "recall": round(micro_r, 6),
            "f1": round(micro_f1, 6),
        },
        "macro_f1_all_configured": round(macro_all, 6),
        "num_configured_labels": len(labels),
        "macro_f1_supported_only": round(macro_sup, 6),
        "num_supported_labels": len(f1_supported),
        "zero_support_labels": [l for l in sorted(labels) if per_class[l]["support"] == 0],
        "per_class": per_class,
    }


def _find_weights_file(model_dir: Path) -> Optional[Path]:
    for fname in ("model.safetensors", "pytorch_model.bin"):
        cand = model_dir / fname
        if cand.exists():
            return cand
    return None


def _load_state_dict(weights_path: Path) -> Dict[str, Any]:
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(weights_path), device="cpu")
    import torch
    return torch.load(str(weights_path), map_location="cpu", weights_only=True)


def _detect_quantized_keys(sd: Dict[str, Any]) -> bool:
    for k in sd.keys():
        if any(marker in k for marker in ("row_scale", "block_scale", "quant_", "log_scale", "alpha_scale")):
            return True
    return False


def extract_checkpoint_step(model_spec: str) -> Optional[int]:
    p = Path(model_spec)
    if p.exists() and p.is_dir():
        ts = p / "trainer_state.json"
        if ts.exists():
            try:
                data = json.loads(ts.read_text(encoding="utf-8"))
                if "global_step" in data:
                    return int(data["global_step"])
            except Exception:
                pass
        if p.name.startswith("checkpoint-"):
            tail = p.name.split("checkpoint-", 1)[-1]
            if tail.isdigit():
                return int(tail)
    return None


def load_ordinary_checkpoint(
    model_spec: str,
    base_model: str,
    device: str,
    max_types: int = 17,
    local_files_only: bool = True,
):
    """
    Load an ordinary (non-quantized) GLiNER model or Trainer checkpoint.
    Refuses to load checkpoints containing quantized buffer/parameter keys.
    """
    from gliner import GLiNER

    p = Path(model_spec)
    if p.exists() and p.is_dir():
        w_file = _find_weights_file(p)
        if w_file is not None:
            sd = _load_state_dict(w_file)
            if _detect_quantized_keys(sd):
                raise RuntimeError(
                    f"Checkpoint {model_spec} contains quantized state_dict keys, "
                    f"but ordinary loader was used. Pass --quantized explicitly."
                )

        if (p / "gliner_config.json").exists():
            model = GLiNER.from_pretrained(str(p), local_files_only=local_files_only)
        else:
            # Trainer intermediate checkpoint without gliner_config.json:
            # instantiate structure from parent (if it has gliner_config.json) or base_model, then load state_dict.
            struct_source = str(p.parent) if (p.parent / "gliner_config.json").exists() else base_model
            model = GLiNER.from_pretrained(struct_source, local_files_only=local_files_only)
            if w_file is None:
                raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {p}")
            sd = _load_state_dict(w_file)
            try:
                model.load_state_dict(sd, strict=True)
            except RuntimeError:
                model.model.load_state_dict(sd, strict=True)
    else:
        model = GLiNER.from_pretrained(model_spec, local_files_only=local_files_only)

    if hasattr(model, "config"):
        model.config.max_types = max_types
    model.to(device)
    model.eval()
    return model


def load_quantized_checkpoint(
    model_spec: str,
    base_model: str,
    device: str,
    scaling_mode: str = "block_max",
    learnable_scaling: bool = True,
    max_types: int = 17,
    local_files_only: bool = True,
):
    """
    Explicitly load a quantized GLiNER checkpoint by constructing the base architecture,
    injecting quantized encoder layers via quantize_gliner_encoder, and loading weights.
    """
    from gliner import GLiNER
    from quantize_gliner import quantize_gliner_encoder

    p = Path(model_spec)
    if not (p.exists() and p.is_dir()):
        raise ValueError(f"Quantized checkpoint must be a local directory: {model_spec}")

    w_file = _find_weights_file(p)
    if w_file is None:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {p}")

    struct_source = str(p) if (p / "gliner_config.json").exists() else (
        str(p.parent) if (p.parent / "gliner_config.json").exists() else base_model
    )
    model = GLiNER.from_pretrained(struct_source, local_files_only=False if not local_files_only else True)
    if hasattr(model, "config"):
        model.config.max_types = max_types

    quantize_gliner_encoder(model, scaling_mode=scaling_mode, learnable=learnable_scaling)
    sd = _load_state_dict(w_file)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.model.load_state_dict(sd, strict=True)

    model.to(device)
    model.eval()
    return model


def build_subword_counter(model):
    """Extract the underlying HuggingFace subword tokenizer from the loaded GLiNER instance."""
    hf_tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if hf_tok is None:
        raise RuntimeError("Could not locate model.data_processor.transformer_tokenizer on GLiNER model.")

    cache: Dict[str, int] = {}

    def count_subwords(word: str) -> int:
        if word in cache:
            return cache[word]
        ids = hf_tok.encode(word, add_special_tokens=False)
        n = max(1, len(ids))
        if len(cache) < 100_000:
            cache[word] = n
        return n

    return count_subwords


def classify_errors(gold: Set[Tuple[int, int, str]], pred: Set[Tuple[int, int, str]]) -> Tuple[Counter, Counter]:
    """Error taxonomy for one window.  Returns (error_counts, fp_in_entity_free_window_by_label)."""
    errs: Counter = Counter()
    ef_by_label: Counter = Counter()

    def overlaps(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    for p_ in pred - gold:
        same_span = [g for g in gold if (g[0], g[1]) == (p_[0], p_[1])]
        ov_same = [g for g in gold if g[2] == p_[2] and overlaps(g, p_)]
        ov_other = [g for g in gold if overlaps(g, p_)]
        if same_span:
            errs["fp_label_confusion"] += 1
        elif ov_same:
            errs["fp_boundary_same_label"] += 1
        elif ov_other:
            errs["fp_overlaps_gold_other_label"] += 1
        elif not gold:
            errs["fp_in_entity_free_window"] += 1
            ef_by_label[p_[2]] += 1
        else:
            errs["fp_spurious_in_positive_window"] += 1
    for g in gold - pred:
        if any((q[0], q[1]) == (g[0], g[1]) for q in pred):
            errs["fn_label_confusion"] += 1
        elif any(q[2] == g[2] and overlaps(g, q) for q in pred):
            errs["fn_boundary"] += 1
        else:
            errs["fn_missed"] += 1
    return errs, ef_by_label


def explain_errors(gold: Set[Tuple[int, int, str]], pred: Set[Tuple[int, int, str]]):
    """Yields (side, kind, span) for every error of one window (same taxonomy as classify_errors)."""
    def overlaps(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    for p_ in sorted(pred - gold):
        if any((g[0], g[1]) == (p_[0], p_[1]) for g in gold):
            kind = "label_confusion"
        elif any(g[2] == p_[2] and overlaps(g, p_) for g in gold):
            kind = "boundary_same_label"
        elif any(overlaps(g, p_) for g in gold):
            kind = "overlaps_gold_other_label"
        elif not gold:
            kind = "in_entity_free_window"
        else:
            kind = "spurious_in_positive_window"
        yield "FP", kind, p_
    for g in sorted(gold - pred):
        if any((q[0], q[1]) == (g[0], g[1]) for q in pred):
            kind = "label_confusion"
        elif any(q[2] == g[2] and overlaps(g, q) for q in pred):
            kind = "boundary"
        else:
            kind = "missed"
        yield "FN", kind, g


_ID_LIKE = re.compile(r"[0-9A-Za-z_-]{12,}")


def _norm_ctx(tokens: Sequence[str]) -> str:
    out = []
    for t in tokens:
        if _ID_LIKE.fullmatch(t) and re.search(r"\d", t):
            out.append("<ID>")
        else:
            out.append(re.sub(r"\d", "0", t))
    return " ".join(out)


def build_error_report(store, threshold: float, top_groups: int = 60, samples_per_group: int = 3):
    """Groups errors by (side, kind, label, left-context) so one repeated template shows up as ONE row.
    Values in this dataset are synthetic stand-ins for the redacted real ones; contexts are traffic text."""
    from collections import defaultdict
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(lambda: {"count": 0, "windows": set(), "samples": []})
    for wi, rec in enumerate(store):
        gold, scored, _ov = rec[0], rec[1], rec[2]
        tokens = rec[3] if len(rec) > 3 else None
        if tokens is None:
            continue
        triples = {(s_, e_, l_) for s_, e_, l_, sc in scored if sc >= threshold}
        score_of = {(s_, e_, l_): sc for s_, e_, l_, sc in scored}
        for side, kind, (s_, e_, l_) in explain_errors(set(gold), triples):
            key = (side, kind, l_, _norm_ctx(tokens[max(0, s_ - 3):s_]))
            g = groups[key]
            g["count"] += 1
            g["windows"].add(wi)
            if len(g["samples"]) < samples_per_group:
                g["samples"].append({
                    "window": wi, "score": round(score_of.get((s_, e_, l_), 0.0), 4),
                    "left": " ".join(tokens[max(0, s_ - 6):s_]), "span": " ".join(tokens[s_:e_ + 1]),
                    "right": " ".join(tokens[e_ + 1:e_ + 4]),
                })
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["count"])
    rows = [{"side": k[0], "kind": k[1], "label": k[2], "left_context": k[3], "count": v["count"],
             "windows": len(v["windows"])} for k, v in ranked[:top_groups]]
    samples = [{"side": k[0], "kind": k[1], "label": k[2], "left_context": k[3], "count": v["count"],
                "examples": v["samples"]} for k, v in ranked[:top_groups]]
    total = sum(v["count"] for v in groups.values())
    top10 = sum(v["count"] for _, v in ranked[:10])
    return rows, samples, {"total_errors": total, "distinct_groups": len(groups), "top10_groups_share": round(top10 / max(1, total), 4)}


def select_threshold(store, grid: Sequence[float], target_positive_rate: Optional[float] = None):
    """Best-micro-F1 threshold over `grid`.  With target_positive_rate, entity-free windows are weighted so
    the validation mix has that share of windows with PII (prior-shift correction: precision depends on
    the mix, and the validation file is much more positive-heavy than real traffic)."""
    n_pos = sum(1 for rec in store if rec[0])
    n_neg = len(store) - n_pos
    w_neg = 1.0
    if target_positive_rate and 0 < target_positive_rate < 1 and n_pos and n_neg:
        w_neg = ((1 - target_positive_rate) / target_positive_rate) / (n_neg / n_pos)
    table = {}
    for t in grid:
        tp = fp = fn = 0.0
        for rec in store:
            gold, scored = set(rec[0]), rec[1]
            pred = {(s_, e_, l_) for s_, e_, l_, sc in scored if sc >= t}
            w = 1.0 if gold else w_neg
            tp += w * len(gold & pred)
            fp += w * len(pred - gold)
            fn += w * len(gold - pred)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        table[f"{t:g}"] = {"precision": round(p, 6), "recall": round(r, 6),
                           "f1": round(2 * p * r / (p + r), 6) if p + r else 0.0}
    best = max(table, key=lambda k: table[k]["f1"])
    return float(best), table, w_neg


def score_store(store, labels: List[str], threshold: float):
    """Score cached (gold, scored_preds, has_overlap) windows at `threshold`."""
    full = {l: Counter() for l in labels}
    clean = {l: Counter() for l in labels}
    errors: Counter = Counter()
    ef_labels: Counter = Counter()
    n_pred_windows_entity_free = 0
    for rec in store:
        gold, scored, has_overlap = rec[0], rec[1], rec[2]
        triples = [(s_, e_, l_) for s_, e_, l_, sc in scored if sc >= threshold]
        ex_stats = match_spans_one_to_one(gold, triples, labels)
        for l in labels:
            full[l].update(ex_stats[l])
            if not has_overlap:
                clean[l].update(ex_stats[l])
        e, ef = classify_errors(set(gold), set(triples))
        errors.update(e)
        ef_labels.update(ef)
        if not gold and triples:
            n_pred_windows_entity_free += 1
    n_ef = sum(1 for rec in store if not rec[0])
    return full, clean, errors, ef_labels, n_ef, n_pred_windows_entity_free


def _is_oom(exc) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def _free_gpu():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _predict_many(model, texts, labels, flat_ner, threshold, multi_label, batch_size, state):
    """Batched inference.  On CUDA OOM the batch is retried one chunk at a time; a chunk that still
    does not fit is counted in state['oom_chunks'] (reported in the audit) instead of crashing the run."""
    if batch_size > 1 and len(texts) > 1 and not state.get("batch_broken"):
        try:
            out = model.inference(texts, labels, flat_ner=flat_ner, threshold=threshold,
                                  multi_label=multi_label, batch_size=batch_size)
            if len(out) == len(texts):
                return out
        except Exception as exc:
            if _is_oom(exc):
                state["batch_oom"] = state.get("batch_oom", 0) + 1
                if state["batch_oom"] <= 3:
                    print("  [note] batch ran out of GPU memory; retrying this batch one chunk at a time", flush=True)
            else:
                state["batch_broken"] = True
                print(f"  [note] batched inference unavailable ({exc!r}); falling back to one window at a time", flush=True)
            _free_gpu()
    results = []
    for t in texts:
        try:
            results.append(model.predict_entities(t, labels, flat_ner=flat_ner, threshold=threshold,
                                                  multi_label=multi_label))
        except Exception as exc:
            if not _is_oom(exc):
                raise
            state["oom_chunks"] = state.get("oom_chunks", 0) + 1
            if state["oom_chunks"] <= 3:
                print("  [warning] a chunk does not fit in GPU memory even alone; counted as no prediction "
                      "(see chunking_audit.chunks_failed_oom)", flush=True)
            _free_gpu()
            results.append([])
    return results


def evaluate_model_on_dataset(
    model,
    parsed_examples: Sequence[Tuple[int, List[str], List[Tuple[int, int, str]], bool]],
    labels: List[str],
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
    allow_enclosing_fallback: bool,
    max_len_override: Optional[int] = None,
    progress_every: int = 1000,
    chunk_unit: str = "words",
    sweep: Optional[Sequence[float]] = None,
    batch_size: int = 1,
    max_subwords: int = 1024,
    dump_errors: bool = False,
    return_store: bool = False,
) -> Dict[str, Any]:
    """Runs the model ONCE over the windows and scores the cached predictions.

    sweep=None  : inference runs at exactly `threshold` -> headline numbers are exact.
    sweep=[...] : inference runs at min(sweep + [threshold]); every threshold is scored by filtering the
                  span scores.  That is exact except for flat_ner's greedy overlap removal, which can
                  differ slightly between thresholds ("headline_is_exact" says which case applies)."""
    import torch

    model_max_len = max_len_override or int(getattr(getattr(model, "config", None), "max_len", 384))
    subword_counter = build_subword_counter(model)
    if chunk_unit == "words":
        chunker = WordChunker(subword_counter, labels=labels, max_len=model_max_len, max_subwords=max_subwords)
    else:
        chunker = SubwordChunker(subword_counter, labels=labels, max_len=model_max_len)

    sweep = sorted(set(float(x) for x in sweep)) if sweep else None
    run_threshold = min(sweep + [threshold]) if sweep else threshold

    chunk_stats = {
        "chunk_unit": chunk_unit,
        "model_max_len": model_max_len,
        "prompt_overhead_subwords": chunker.prompt_overhead,
        "text_budget_unit_budget": chunker.text_budget,
        "total_examples_evaluated": len(parsed_examples),
        "single_chunk_examples": 0,
        "multi_chunk_examples": 0,
        "total_chunks_executed": 0,
        "max_window_subwords_seen": 0,
        "windows_exceeding_max_len_before_chunking": 0,
        "single_tokens_exceeding_budget": 0,
        "raw_chunk_predictions": 0,
        "deduplicated_predictions": 0,
        "duplicates_removed_across_chunks": 0,
        "exact_token_boundary_predictions": 0,
        "subtoken_boundary_mismatches": 0,
        "run_threshold": run_threshold,
    }

    store = []
    state: Dict[str, Any] = {}
    pending = []   # [(has_overlap, gold, [ (chunk, offsets, start_map, end_map, text) ], tokens)]

    def flush():
        if not pending:
            return
        texts = [c[4] for _, _, chs, _t in pending for c in chs]
        preds_all = _predict_many(model, texts, labels, flat_ner, run_threshold, multi_label, batch_size, state)
        k = 0
        for has_overlap, gold, chs, win_tokens in pending:
            window_raw: List[Tuple[int, int, str, float]] = []
            for (ch, offsets, start_map, end_map, _t) in chs:
                for ent in preds_all[k]:
                    chunk_stats["raw_chunk_predictions"] += 1
                    mapped, is_exact = map_char_pred_to_tokens(
                        int(ent["start"]), int(ent["end"]), offsets, start_map, end_map,
                        allow_enclosing_fallback=allow_enclosing_fallback)
                    if mapped is None:
                        chunk_stats["subtoken_boundary_mismatches"] += 1
                        continue
                    if is_exact:
                        chunk_stats["exact_token_boundary_predictions"] += 1
                    else:
                        chunk_stats["subtoken_boundary_mismatches"] += 1
                    window_raw.append((ch.start_tok + mapped[0], ch.start_tok + mapped[1],
                                       str(ent["label"]), float(ent.get("score", 1.0))))
                k += 1
            dedup = deduplicate_predictions(window_raw)
            chunk_stats["deduplicated_predictions"] += len(dedup)
            chunk_stats["duplicates_removed_across_chunks"] += len(window_raw) - len(dedup)
            store.append((gold, dedup, has_overlap, win_tokens))
        pending.clear()

    t0 = time.time()
    with torch.inference_mode():
        for idx_pos, (_line_idx, tokens, gold_spans, has_overlap) in enumerate(parsed_examples, 1):
            chunks, word_subwords = chunker.chunk_tokens(tokens)
            win_total_sub = chunker.prompt_overhead + sum(word_subwords)
            chunk_stats["max_window_subwords_seen"] = max(chunk_stats["max_window_subwords_seen"], win_total_sub)
            if win_total_sub > model_max_len:
                chunk_stats["windows_exceeding_max_len_before_chunking"] += 1
            chunk_stats["single_chunk_examples" if len(chunks) == 1 else "multi_chunk_examples"] += 1
            chunk_stats["total_chunks_executed"] += len(chunks)

            chs = []
            for ch in chunks:
                if getattr(ch, "exceeded_single_token_budget", False):
                    chunk_stats["single_tokens_exceeding_budget"] += 1
                ch_text, offsets, start_map, end_map = build_chunk_string_and_offsets(tokens[ch.start_tok: ch.end_tok + 1])
                chs.append((ch, offsets, start_map, end_map, ch_text))
            pending.append((has_overlap, gold_spans, chs, tokens))
            if len(pending) >= max(1, batch_size):
                flush()

            if progress_every > 0 and idx_pos % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  [{idx_pos}/{len(parsed_examples)}] ({100.0 * idx_pos / len(parsed_examples):.1f}%) "
                      f"| {idx_pos / max(1e-6, elapsed):.1f} win/s | elapsed {elapsed:.1f}s", flush=True)
        flush()

    elapsed_total = time.time() - t0
    chunk_stats["chunks_failed_oom"] = state.get("oom_chunks", 0)
    chunk_stats["batches_retried_after_oom"] = state.get("batch_oom", 0)
    if chunk_stats["chunks_failed_oom"]:
        print(f"WARNING: {chunk_stats['chunks_failed_oom']} chunk(s) could not be run (GPU OOM); "
              f"they count as misses.  Lower --max-subwords.", flush=True)
    full, clean, errors, ef_labels, n_ef, n_ef_with_pred = score_store(store, labels, threshold)
    out = {
        "elapsed_seconds": round(elapsed_total, 2),
        "windows_per_second": round(len(parsed_examples) / max(1e-6, elapsed_total), 2),
        "headline_threshold": threshold,
        "headline_is_exact": sweep is None or abs(run_threshold - threshold) < 1e-12,
        "chunking_audit": chunk_stats,
        "metrics_full_dataset": compute_metrics_table(full, labels),
        "metrics_clean_subset_no_overlapping_gold": compute_metrics_table(clean, labels),
        "error_breakdown": dict(errors),
        "false_positives_in_entity_free_windows_by_label": dict(ef_labels.most_common()),
        "entity_free_windows": n_ef,
        "entity_free_windows_with_a_prediction": n_ef_with_pred,
    }
    if dump_errors:
        rows, samples, summ = build_error_report(store, threshold)
        out["error_groups"], out["error_samples"], out["error_groups_summary"] = rows, samples, summ
    if return_store:
        out["_store"] = store
    if sweep:
        table = {}
        for t in sweep:
            f_, _c, _e, _l, _n, _w = score_store(store, labels, t)
            m = compute_metrics_table(f_, labels)
            table[f"{t:g}"] = {"micro_precision": m["micro"]["precision"], "micro_recall": m["micro"]["recall"],
                               "micro_f1": m["micro"]["f1"], "macro_f1_supported": m["macro_f1_supported_only"],
                               "tp": m["micro"]["tp"], "fp": m["micro"]["fp"], "fn": m["micro"]["fn"]}
        out["threshold_sweep"] = table
    return out


def write_per_class_csv(csv_path: Path, model_name: str, step: Optional[int], metrics_dict: Dict[str, Any]):
    fieldnames = [
        "model",
        "checkpoint_step",
        "label",
        "support",
        "predicted",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "has_support",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in metrics_dict["per_class"].items():
            writer.writerow(
                {
                    "model": model_name,
                    "checkpoint_step": "" if step is None else step,
                    "label": label,
                    **row,
                }
            )


def sanitize_tag(model_spec: str) -> str:
    p = Path(model_spec)
    if p.exists():
        if p.name.startswith("checkpoint-"):
            return f"{p.parent.name}_{p.name}"
        return p.name
    return model_spec.replace("/", "__").replace("\\", "__").replace(":", "")


def find_training_files(model_spec: str) -> Optional[Path]:
    """Folder holding train_labels.json / training_meta.json for a model path (the model folder,
    or the output folder that contains the checkpoint-XXXX / best_checkpoint folder)."""
    p = Path(model_spec)
    if not p.exists():
        return None
    for cand in (p, p.parent):
        if (cand / "train_labels.json").exists():
            return cand
    return None


def resolve_labels(model_spec: str, labels_file: Optional[str]) -> Tuple[List[str], str]:
    if labels_file:
        return sorted(json.loads(Path(labels_file).read_text(encoding="utf-8"))), f"--labels-file {labels_file}"
    d = find_training_files(model_spec)
    if d is not None:
        return sorted(json.loads((d / "train_labels.json").read_text(encoding="utf-8"))), f"{d / 'train_labels.json'}"
    return get_configured_labels(), "ALL 17 configured labels (no train_labels.json found: untrained labels can only add false positives)"


def parse_dataset(data_path: Path, allowed_labels: Set[str], limit: Optional[int]):
    summary = ValidationSummary()
    parsed: List[Tuple[int, List[str], List[Tuple[int, int, str]], bool]] = []
    with open(data_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                raw_obj = json.loads(line)
            except json.JSONDecodeError:
                summary.total_lines += 1
                summary.malformed_examples += 1
                summary.invalid_example_indices.append(line_idx)
                continue
            res = validate_and_parse_example(raw_obj, line_idx, allowed_labels, summary)
            if res is not None:
                tokens, dedup_spans, overlaps = res
                parsed.append((line_idx, tokens, dedup_spans, overlaps))
            if limit is not None and len(parsed) >= limit:
                break
    return summary, parsed


def main():
    ap = argparse.ArgumentParser(description="Audit-grade evaluator for AntShield GLiNER models.")
    ap.add_argument("--models", nargs="+", help="One or more model paths or HF hub IDs to evaluate on the exact same examples.")
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1", help="Base model architecture reference.")
    ap.add_argument("--data-path", default="data/test_finetune.jsonl", help="JSONL dataset to evaluate (the TEST file).")
    ap.add_argument("--val-path", default=None,
                    help="VALIDATION jsonl (data/val_finetune.jsonl).  When given, the confidence threshold is "
                         "selected on it (best micro-F1) and then applied once, exactly, to --data-path.")
    ap.add_argument("--labels-file", default=None,
                    help="JSON list of the labels to prompt with (default: train_labels.json next to the model).")
    ap.add_argument("--output-dir", default="eval_audit_results", help="Directory for JSON/CSV reports.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed confidence threshold.  Default: selected on --val-path if given, else 0.5.")
    ap.add_argument("--sweep", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="Thresholds for the validation sweep (comma separated).")
    ap.add_argument("--target-positive-rate", type=float, default=None,
                    help="Share of windows that contain PII in the deployment you care about (e.g. 0.20). When given, "
                         "entity-free validation windows are re-weighted to that mix before the threshold is chosen.")
    ap.add_argument("--dump-errors", action="store_true",
                    help="Write <tag>_error_groups.csv and <tag>_error_samples.json: errors grouped by label and "
                         "left context, with a few examples each.")
    ap.add_argument("--sweep-test", action="store_true",
                    help="Also report a threshold sweep on the test file (for transparency only; never used for selection).")
    ap.add_argument("--max-types", type=int, default=25, help="model.config.max_types (must be >= number of labels).")
    ap.add_argument("--max-len", type=int, default=None, help="Override model max_len (default: model.config.max_len, in WORDS).")
    ap.add_argument("--chunk-unit", choices=["words", "subwords"], default="words",
                    help="'words' (default, GLiNER's own unit) or the legacy 'subwords' budget.")
    ap.add_argument("--batch-size", type=int, default=4, help="windows per forward pass (1 = one at a time).")
    ap.add_argument("--max-subwords", type=int, default=1024,
                    help="max subwords per forward pass incl. the label prompt (training dropped windows above 1024). "
                         "Lower it (e.g. 768) if you still run out of GPU memory.")
    ap.add_argument("--no-flat-ner", dest="flat_ner", action="store_false", help="Disable flat_ner in predict_entities.")
    ap.add_argument("--multi-label", action="store_true", help="Enable multi_label in predict_entities.")
    ap.add_argument("--allow-enclosing-boundary", action="store_true", help="Snap sub-token char spans to enclosing token.")
    ap.add_argument("--quantized", action="store_true", help="Load checkpoints using the quantized loader (NEVER default).")
    ap.add_argument("--scaling-mode", default="block_max", choices=["block_max", "row_sigma", "row_sigma_learnable"])
    ap.add_argument("--no-learnable-scaling", dest="learnable_scaling", action="store_false")
    ap.add_argument("--device", default=None, help="Device (cuda or cpu). Default: cuda if available else cpu.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples for quick smoke check.")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing report files in output-dir.")
    ap.add_argument("--run-tests", action="store_true", help="Run built-in synthetic unit test suite and exit.")
    args = ap.parse_args()

    if args.run_tests:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent), pattern="test_evaluate_gliner_audit.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if not args.models:
        ap.error("--models is required unless --run-tests is specified.")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    all_known = set(get_configured_labels())
    sweep_grid = [float(x) for x in args.sweep.split(",") if x.strip()]

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    print(f"Auditing dataset: {data_path} ...")
    dataset_sha256 = sha256_file(data_path)
    val_summary, parsed_examples = parse_dataset(data_path, all_known, args.limit)
    print(
        f"Dataset validation complete: {val_summary.valid_examples}/{val_summary.total_lines} valid examples | "
        f"unique gold spans={val_summary.valid_unique_gold_spans} | "
        f"duplicate gold spans={val_summary.duplicate_gold_spans} | "
        f"invalid gold spans={val_summary.invalid_gold_spans} | "
        f"examples with overlapping gold spans={val_summary.examples_with_overlapping_gold}"
    )
    val_parsed, val_sha = None, None
    if args.val_path:
        vp = Path(args.val_path)
        if not vp.exists():
            raise FileNotFoundError(f"Validation file not found: {vp}")
        val_sha = sha256_file(vp)
        if val_sha == dataset_sha256:
            raise ValueError("--val-path and --data-path are the same file; threshold would be tuned on the test set.")
        _vs, val_parsed = parse_dataset(vp, all_known, args.limit)
        print(f"Validation file: {len(val_parsed)} windows ({sum(1 for x in val_parsed if x[2])} with gold spans)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for model_spec in args.models:
        tag = sanitize_tag(model_spec)
        json_out = out_dir / f"{tag}_audit_report.json"
        csv_out = out_dir / f"{tag}_per_class_metrics.csv"
        if not args.overwrite and (json_out.exists() or csv_out.exists()):
            raise FileExistsError(
                f"Output file {json_out} or {csv_out} already exists. Pass --overwrite or a new --output-dir."
            )

        step = extract_checkpoint_step(model_spec)
        loader_type = "quantized" if args.quantized else "ordinary"
        labels, labels_source = resolve_labels(model_spec, args.labels_file)
        max_types = max(args.max_types, len(labels))
        print(f"\n==== Evaluating [{tag}] (step={step}, loader={loader_type}, device={device}) ====")
        print(f"Prompt labels ({len(labels)}): from {labels_source}")

        # contamination check against the training run's own record (replaces the old unconditional warning)
        contamination = {"checked": False}
        tdir = find_training_files(model_spec)
        if tdir is not None and (tdir / "training_meta.json").exists():
            try:
                meta = json.loads((tdir / "training_meta.json").read_text(encoding="utf-8"))
                used = meta.get("data_sha256", {})
                hit = [n for n, h in used.items() if h == dataset_sha256]
                contamination = {"checked": True, "evaluated_file_was_used_in_training_run": bool(hit), "files": hit}
                if hit:
                    print(f"WARNING: {data_path} is the file this model was trained/validated on ({hit}); "
                          f"these numbers are NOT held-out test numbers.")
            except Exception as exc:
                contamination = {"checked": False, "error": repr(exc)}
        unknown_gold = Counter()
        for _i, _t, g, _o in parsed_examples:
            for _s, _e, l in g:
                if l not in labels:
                    unknown_gold[l] += 1
        if unknown_gold:
            print(f"WARNING: {sum(unknown_gold.values())} gold spans have labels the model is not prompted with "
                  f"{dict(unknown_gold)}; they count as misses only if you add the label via --labels-file.")

        if args.quantized:
            model = load_quantized_checkpoint(
                model_spec, base_model=args.base_model, device=device,
                scaling_mode=args.scaling_mode, learnable_scaling=args.learnable_scaling, max_types=max_types,
            )
        else:
            model = load_ordinary_checkpoint(model_spec, base_model=args.base_model, device=device, max_types=max_types)

        common = dict(
            model=model, labels=labels, flat_ner=args.flat_ner, multi_label=args.multi_label,
            allow_enclosing_fallback=args.allow_enclosing_boundary, max_len_override=args.max_len,
            progress_every=args.progress_every, chunk_unit=args.chunk_unit, batch_size=args.batch_size,
            max_subwords=args.max_subwords,
        )

        threshold_source = "fixed (--threshold)"
        val_block = None
        if args.threshold is not None:
            threshold = args.threshold
        elif val_parsed is not None:
            print("Selecting the confidence threshold on the VALIDATION file ...")
            v = evaluate_model_on_dataset(parsed_examples=val_parsed, threshold=0.5, sweep=sweep_grid,
                                          return_store=True, **common)
            val_store = v.pop("_store")
            best_t, best_f1 = None, -1.0
            for t_str, row in v["threshold_sweep"].items():
                if row["micro_f1"] > best_f1:
                    best_t, best_f1 = float(t_str), row["micro_f1"]
            weighted_info = None
            if args.target_positive_rate:
                best_t, wtable, w_neg = select_threshold(val_store, sweep_grid, args.target_positive_rate)
                best_f1 = wtable[f"{best_t:g}"]["f1"]
                weighted_info = {"target_positive_rate": args.target_positive_rate, "entity_free_weight": round(w_neg, 4),
                                 "weighted_table": wtable}
                print(f"  prior-shift weighting: entity-free val windows x{w_neg:.2f} "
                      f"(target {args.target_positive_rate:.0%} windows with PII)")
            threshold = best_t
            how = "weighted " if weighted_info else ""
            threshold_source = f"selected on {args.val_path} (best {how}micro-F1 {best_f1:.4f})"
            if threshold in (min(sweep_grid), max(sweep_grid)):
                print(f"  WARNING: selected threshold {threshold:g} is at the edge of the sweep grid; "
                      f"the optimum may lie outside it (extend --sweep).")
            val_block = {"val_path": args.val_path, "val_sha256": val_sha, "selected_threshold": threshold,
                         "prior_shift": weighted_info,
                         "sweep": v["threshold_sweep"], "error_breakdown": v["error_breakdown"],
                         "entity_free_windows": v["entity_free_windows"],
                         "entity_free_windows_with_a_prediction": v["entity_free_windows_with_a_prediction"]}
            print(f"  selected threshold = {threshold:g} (val micro-F1 {best_f1:.4f})")
        else:
            threshold = 0.5
            threshold_source = "default 0.5 (no --val-path given; pass one to select the threshold properly)"

        eval_out = evaluate_model_on_dataset(
            parsed_examples=parsed_examples, threshold=threshold,
            sweep=(sweep_grid if args.sweep_test else None), dump_errors=args.dump_errors, **common)

        full_m = eval_out["metrics_full_dataset"]
        clean_m = eval_out["metrics_clean_subset_no_overlapping_gold"]

        report = {
            "model_path": model_spec,
            "model_tag": tag,
            "checkpoint_step": step,
            "loader_type": loader_type,
            "base_model": args.base_model,
            "evaluation_settings": {
                "threshold": threshold,
                "threshold_source": threshold_source,
                "flat_ner": args.flat_ner,
                "multi_label": args.multi_label,
                "allow_enclosing_boundary": args.allow_enclosing_boundary,
                "max_types": max_types,
                "chunk_unit": args.chunk_unit,
                "batch_size": args.batch_size,
                "max_subwords": args.max_subwords,
                "device": device,
                "limit": args.limit,
                "labels_source": labels_source,
            },
            "dataset_audit": {
                "data_path": str(data_path),
                "dataset_sha256": dataset_sha256,
                "total_lines": val_summary.total_lines,
                "valid_examples": val_summary.valid_examples,
                "malformed_examples": val_summary.malformed_examples,
                "total_raw_gold_spans": val_summary.total_raw_gold_spans,
                "valid_unique_gold_spans": val_summary.valid_unique_gold_spans,
                "duplicate_gold_spans": val_summary.duplicate_gold_spans,
                "invalid_gold_spans": val_summary.invalid_gold_spans,
                "unknown_label_spans": val_summary.unknown_label_spans,
                "examples_with_overlapping_gold": val_summary.examples_with_overlapping_gold,
                "overlapping_gold_span_count": val_summary.overlapping_gold_span_count,
                "invalid_example_indices_sample": val_summary.invalid_example_indices[:50],
                "overlapping_example_indices_sample": val_summary.overlapping_example_indices[:100],
            },
            "training_run_overlap_check": contamination,
            "validation_threshold_selection": val_block,
            "configured_labels": labels,
            **eval_out,
        }

        if args.dump_errors:
            samples = report.pop("error_samples")
            (out_dir / f"{tag}_error_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
            with open(out_dir / f"{tag}_error_groups.csv", "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=["side", "kind", "label", "left_context", "count", "windows"])
                wr.writeheader()
                wr.writerows(report["error_groups"])
            sm = report["error_groups_summary"]
            print(f"  error groups: {sm['distinct_groups']} distinct, top-10 groups = {sm['top10_groups_share']:.0%} of "
                  f"{sm['total_errors']} errors -> {tag}_error_groups.csv / _error_samples.json")
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_per_class_csv(csv_out, model_spec, step, full_m)
        if eval_out.get("threshold_sweep"):
            print("  test threshold sweep (for understanding only, never used for selection):")
            print("    thr     P       R       F1")
            for t_str, row in eval_out["threshold_sweep"].items():
                print(f"    {t_str:<6} {row['micro_precision']:.4f}  {row['micro_recall']:.4f}  {row['micro_f1']:.4f}")

        print(
            f"Summary [{tag}] @thr={threshold:g}: Micro-P={full_m['micro']['precision']:.4f} "
            f"Micro-R={full_m['micro']['recall']:.4f} "
            f"Micro-F1={full_m['micro']['f1']:.4f} | "
            f"Macro-F1 ({full_m['num_configured_labels']} prompted labels)={full_m['macro_f1_all_configured']:.4f} | "
            f"Macro-F1 (with test support, {full_m['num_supported_labels']})={full_m['macro_f1_supported_only']:.4f}"
        )
        print(f"  errors: {eval_out['error_breakdown']}")
        print(f"  entity-free windows with >=1 prediction: {eval_out['entity_free_windows_with_a_prediction']}/{eval_out['entity_free_windows']}")
        print(f"Saved JSON -> {json_out}")
        print(f"Saved CSV  -> {csv_out}")

        summary_rows.append(
            {
                "model_tag": tag,
                "model_path": model_spec,
                "checkpoint_step": "" if step is None else step,
                "loader_type": loader_type,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "micro_precision": full_m["micro"]["precision"],
                "micro_recall": full_m["micro"]["recall"],
                "micro_f1": full_m["micro"]["f1"],
                "macro_f1_all_prompted": full_m["macro_f1_all_configured"],
                "macro_f1_supported": full_m["macro_f1_supported_only"],
                "clean_subset_micro_f1": clean_m["micro"]["f1"],
                "clean_subset_macro_f1_supported": clean_m["macro_f1_supported_only"],
                "tp": full_m["micro"]["tp"],
                "fp": full_m["micro"]["fp"],
                "fn": full_m["micro"]["fn"],
                "support": full_m["micro"]["support"],
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(summary_rows) > 0:
        comp_csv = out_dir / "summary_comparison.csv"
        comp_json = out_dir / "summary_comparison.json"
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        comp_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        print(f"\nWrote multi-model comparison -> {comp_csv}")


if __name__ == "__main__":
    main()
