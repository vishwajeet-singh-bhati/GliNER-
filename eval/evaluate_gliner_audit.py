#!/usr/bin/env python3
"""
Audit-grade GLiNER Evaluator (eval/evaluate_gliner_audit.py)
Read-only evaluation of base, fine-tuned, and quantized GLiNER checkpoints.
Never modifies existing datasets or checkpoints. Never logs raw PII text.
"""
import argparse
import csv
import hashlib
import json
import os
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
) -> Dict[str, Any]:
    import torch

    model_max_len = max_len_override or int(getattr(getattr(model, "config", None), "max_len", 384))
    subword_counter = build_subword_counter(model)
    chunker = SubwordChunker(subword_counter, labels=labels, max_len=model_max_len)

    full_totals = {l: Counter() for l in labels}
    clean_totals = {l: Counter() for l in labels}

    chunk_stats = {
        "model_max_len": model_max_len,
        "prompt_overhead_subwords": chunker.prompt_overhead,
        "text_subword_budget": chunker.text_budget,
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
    }

    t0 = time.time()
    with torch.inference_mode():
        for idx_pos, (_line_idx, tokens, gold_spans, has_overlap) in enumerate(parsed_examples, 1):
            chunks, word_subwords = chunker.chunk_tokens(tokens)
            win_total_sub = chunker.prompt_overhead + sum(word_subwords)
            if win_total_sub > chunk_stats["max_window_subwords_seen"]:
                chunk_stats["max_window_subwords_seen"] = win_total_sub
            if win_total_sub > model_max_len:
                chunk_stats["windows_exceeding_max_len_before_chunking"] += 1

            if len(chunks) == 1:
                chunk_stats["single_chunk_examples"] += 1
            else:
                chunk_stats["multi_chunk_examples"] += 1

            chunk_stats["total_chunks_executed"] += len(chunks)

            window_raw_preds: List[Tuple[int, int, str, float]] = []
            for ch in chunks:
                if ch.exceeded_single_token_budget:
                    chunk_stats["single_tokens_exceeding_budget"] += 1
                ch_tokens = tokens[ch.start_tok : ch.end_tok + 1]
                ch_text, offsets, start_map, end_map = build_chunk_string_and_offsets(ch_tokens)

                raw_ents = model.predict_entities(
                    ch_text,
                    labels,
                    flat_ner=flat_ner,
                    threshold=threshold,
                    multi_label=multi_label,
                )
                for ent in raw_ents:
                    chunk_stats["raw_chunk_predictions"] += 1
                    mapped, is_exact = map_char_pred_to_tokens(
                        int(ent["start"]),
                        int(ent["end"]),
                        offsets,
                        start_map,
                        end_map,
                        allow_enclosing_fallback=allow_enclosing_fallback,
                    )
                    if mapped is None:
                        chunk_stats["subtoken_boundary_mismatches"] += 1
                        continue
                    if is_exact:
                        chunk_stats["exact_token_boundary_predictions"] += 1
                    else:
                        chunk_stats["subtoken_boundary_mismatches"] += 1

                    g_start = ch.start_tok + mapped[0]
                    g_end = ch.start_tok + mapped[1]
                    window_raw_preds.append((g_start, g_end, str(ent["label"]), float(ent.get("score", 1.0))))

            dedup_preds_scored = deduplicate_predictions(window_raw_preds)
            chunk_stats["deduplicated_predictions"] += len(dedup_preds_scored)
            chunk_stats["duplicates_removed_across_chunks"] += len(window_raw_preds) - len(dedup_preds_scored)

            pred_triples = [(s, e, l) for s, e, l, _ in dedup_preds_scored]
            ex_stats = match_spans_one_to_one(gold_spans, pred_triples, labels)

            for l in labels:
                full_totals[l].update(ex_stats[l])
                if not has_overlap:
                    clean_totals[l].update(ex_stats[l])

            if progress_every > 0 and idx_pos % progress_every == 0:
                elapsed = time.time() - t0
                rate = idx_pos / max(1e-6, elapsed)
                print(
                    f"  [{idx_pos}/{len(parsed_examples)}] ({100.0 * idx_pos / len(parsed_examples):.1f}%) "
                    f"| {rate:.1f} win/s | elapsed {elapsed:.1f}s",
                    flush=True,
                )

    elapsed_total = time.time() - t0
    return {
        "elapsed_seconds": round(elapsed_total, 2),
        "windows_per_second": round(len(parsed_examples) / max(1e-6, elapsed_total), 2),
        "chunking_audit": chunk_stats,
        "metrics_full_dataset": compute_metrics_table(full_totals, labels),
        "metrics_clean_subset_no_overlapping_gold": compute_metrics_table(clean_totals, labels),
    }


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


def main():
    ap = argparse.ArgumentParser(description="Audit-grade evaluator for AntShield GLiNER models.")
    ap.add_argument(
        "--models",
        nargs="+",
        help="One or more model paths or HF hub IDs to evaluate on the exact same examples.",
    )
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1", help="Base model architecture reference.")
    ap.add_argument("--data-path", default="data/test_finetune.jsonl", help="JSONL dataset to evaluate.")
    ap.add_argument("--output-dir", default="D:/models/eval_audit_results", help="Directory for JSON/CSV reports.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Entity confidence threshold.")
    ap.add_argument("--max-types", type=int, default=17, help="model.config.max_types setting.")
    ap.add_argument("--max-len", type=int, default=None, help="Override model max_len for chunking (default: model.config.max_len).")
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
    labels = get_configured_labels()
    allowed_labels = set(labels)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    print(f"Auditing dataset: {data_path} ...")
    dataset_sha256 = sha256_file(data_path)
    val_summary = ValidationSummary()
    parsed_examples: List[Tuple[int, List[str], List[Tuple[int, int, str]], bool]] = []

    with open(data_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                raw_obj = json.loads(line)
            except json.JSONDecodeError:
                val_summary.total_lines += 1
                val_summary.malformed_examples += 1
                val_summary.invalid_example_indices.append(line_idx)
                continue

            res = validate_and_parse_example(raw_obj, line_idx, allowed_labels, val_summary)
            if res is not None:
                tokens, dedup_spans, overlaps = res
                parsed_examples.append((line_idx, tokens, dedup_spans, overlaps))
            if args.limit is not None and len(parsed_examples) >= args.limit:
                break

    print(
        f"Dataset validation complete: {val_summary.valid_examples}/{val_summary.total_lines} valid examples | "
        f"unique gold spans={val_summary.valid_unique_gold_spans} | "
        f"duplicate gold spans={val_summary.duplicate_gold_spans} | "
        f"invalid gold spans={val_summary.invalid_gold_spans} | "
        f"examples with overlapping gold spans={val_summary.examples_with_overlapping_gold}"
    )
    print(
        "CONTAMINATION WARNING: train/finetune_gliner.py currently passes test_finetune.jsonl as eval_dataset. "
        "Do NOT select or tune checkpoints using test_finetune.jsonl metrics."
    )

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
        print(f"\n==== Evaluating [{tag}] (step={step}, loader={loader_type}, device={device}) ====")

        if args.quantized:
            model = load_quantized_checkpoint(
                model_spec,
                base_model=args.base_model,
                device=device,
                scaling_mode=args.scaling_mode,
                learnable_scaling=args.learnable_scaling,
                max_types=args.max_types,
            )
        else:
            model = load_ordinary_checkpoint(
                model_spec,
                base_model=args.base_model,
                device=device,
                max_types=args.max_types,
            )

        eval_out = evaluate_model_on_dataset(
            model=model,
            parsed_examples=parsed_examples,
            labels=labels,
            threshold=args.threshold,
            flat_ner=args.flat_ner,
            multi_label=args.multi_label,
            allow_enclosing_fallback=args.allow_enclosing_boundary,
            max_len_override=args.max_len,
            progress_every=args.progress_every,
        )

        full_m = eval_out["metrics_full_dataset"]
        clean_m = eval_out["metrics_clean_subset_no_overlapping_gold"]

        report = {
            "model_path": model_spec,
            "model_tag": tag,
            "checkpoint_step": step,
            "loader_type": loader_type,
            "base_model": args.base_model,
            "evaluation_settings": {
                "threshold": args.threshold,
                "flat_ner": args.flat_ner,
                "multi_label": args.multi_label,
                "allow_enclosing_boundary": args.allow_enclosing_boundary,
                "max_types": args.max_types,
                "device": device,
                "limit": args.limit,
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
            "test_set_contamination_warning": {
                "contaminated_eval_split_in_training_script": True,
                "detail": (
                    "train/finetune_gliner.py sets val_data = test_finetune.jsonl. "
                    "Do not select the best checkpoint using test_finetune.jsonl."
                ),
            },
            "configured_labels": labels,
            **eval_out,
        }

        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_per_class_csv(csv_out, model_spec, step, full_m)

        print(
            f"Summary [{tag}]: Micro-P={full_m['micro']['precision']:.4f} "
            f"Micro-R={full_m['micro']['recall']:.4f} "
            f"Micro-F1={full_m['micro']['f1']:.4f} | "
            f"Macro-F1 (17 labels)={full_m['macro_f1_all_configured']:.4f} | "
            f"Macro-F1 (16 supported)={full_m['macro_f1_supported_only']:.4f}"
        )
        print(f"Saved JSON -> {json_out}")
        print(f"Saved CSV  -> {csv_out}")

        summary_rows.append(
            {
                "model_tag": tag,
                "model_path": model_spec,
                "checkpoint_step": "" if step is None else step,
                "loader_type": loader_type,
                "threshold": args.threshold,
                "micro_precision": full_m["micro"]["precision"],
                "micro_recall": full_m["micro"]["recall"],
                "micro_f1": full_m["micro"]["f1"],
                "macro_f1_all_17": full_m["macro_f1_all_configured"],
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
