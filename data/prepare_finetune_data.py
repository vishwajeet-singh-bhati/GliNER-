"""
Builds GLiNER fine-tuning data from the AntShield dataset:
  1. loads + cleans records (same header-whitelist cleaning as eval path)
  2. finds RECON_<Type> spans, optionally masks them with synthetic values
     (same logic as convert_to_gliner_format.py -- see that file's docstring
     for why masking matters)
  3. tokenizes with GLiNER's own tokenization regex and converts char spans
     to token spans (data/gliner_tokenizer.py)
  4. chunks any record whose token count exceeds --max-tokens into
     overlapping windows, so training never silently truncates/drops entities
  5. splits deterministically into train/val/test by hashing record_id (so
     rerunning is reproducible, and adding new data later doesn't reshuffle
     existing splits)

Output: three JSONL files, each line {"tokenized_text": [...], "ner": [[s,e,label],...]}
-- this is exactly the format GLiNER's model.train_model expects (once loaded
back into a Python list, see train/finetune_gliner.py).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_type_mapping import build_recon_pattern, extract_recon_label  # noqa: E402
from antshield_loader import load_antshield_dataset, PacketRecord  # noqa: E402
from recon_synthesizer import synthesize  # noqa: E402
from gliner_tokenizer import tokenize_with_offsets, char_span_to_token_span  # noqa: E402
from splitting import assign_split, assign_split_campaign  # noqa: E402

_RECON_PATTERN = build_recon_pattern()


def build_char_entities(text: str, mask_recon_tokens: bool, seed_prefix: str):
    """Same span-finding/masking logic as convert_to_gliner_format.py,
    returning (final_text, [(char_start, char_end, gliner_label), ...])."""
    matches = list(_RECON_PATTERN.finditer(text))

    if not mask_recon_tokens:
        entities = [
            (m.start(), m.end(), extract_recon_label(m.group(0))) for m in matches
        ]
        return text, entities

    out_parts = []
    entities = []
    cursor = 0
    for i, m in enumerate(matches):
        out_parts.append(text[cursor:m.start()])
        label = extract_recon_label(m.group(0))
        replacement = synthesize(label, seed_key=f"{seed_prefix}:{i}")
        new_start = sum(len(p) for p in out_parts)
        out_parts.append(replacement)
        new_end = new_start + len(replacement)
        entities.append((new_start, new_end, label))
        cursor = m.end()
    out_parts.append(text[cursor:])
    return "".join(out_parts), entities


def chunk_and_tokenize(text: str, char_entities, max_tokens: int, overlap_tokens: int, max_windows_per_record: int = 3):
    """Yields {"tokenized_text": [...], "ner": [[s,e,label],...]} dicts, one
    per window, for windows that fit within max_tokens. Entities that don't
    align to clean token boundaries, or that straddle a window boundary, are
    dropped from that window (logged by the caller via return counts).

    max_windows_per_record caps runaway chunking from a small number of
    extreme outliers -- confirmed in this dataset: 1,050 records (2.2%) are
    crash-report/stack-trace JSON blobs exceeding 2,000 tokens each (one
    exceeds 16,000), which without a cap would generate ~50+ largely-identical
    windows per record and dominate the training set with low-value content.
    Windows containing an entity are kept first, then earliest windows fill
    any remaining slots, so a genuine PII example in a long record is never
    the one that gets dropped."""
    tokens = tokenize_with_offsets(text)
    if not tokens:
        return [{"tokenized_text": [], "ner": []}], 0

    dropped = 0
    windows = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        window_tokens = tokens[start:end]
        window_char_start = window_tokens[0][1]
        window_char_end = window_tokens[-1][2]

        ner = []
        for c_start, c_end, label in char_entities:
            if c_start < window_char_start or c_end > window_char_end:
                continue  # entity outside this window, or straddles its edge
            span = char_span_to_token_span(window_tokens, c_start, c_end)
            if span is None:
                dropped += 1
                continue
            # span indices are relative to window_tokens already since we
            # passed the sliced list
            ner.append([span[0], span[1], label])

        windows.append({"tokenized_text": [t[0] for t in window_tokens], "ner": ner})

        if end == len(tokens):
            break
        start = end - overlap_tokens

    if len(windows) > max_windows_per_record:
        with_entities = [w for w in windows if w["ner"]]
        without_entities = [w for w in windows if not w["ner"]]
        keep = with_entities[:max_windows_per_record]
        remaining_slots = max_windows_per_record - len(keep)
        keep.extend(without_entities[:remaining_slots])
        windows = keep

    return windows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--mask-recon-tokens", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=300, help="window size, kept under GLiNER's 384 limit")
    ap.add_argument("--overlap-tokens", type=int, default=30)
    ap.add_argument("--max-windows-per-record", type=int, default=3)
    ap.add_argument(
        "--split-strategy",
        choices=["campaign", "hash"],
        default="campaign",
        help="'campaign' (default): auto_anteater=train (minus a val "
             "carve-out), manual_anteater=test -- matches the reference "
             "binary-BERT model's split. 'hash': random per-record split.",
    )
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument(
        "--include-metadata-fields",
        action="store_true",
        help="see convert_to_gliner_format.py --help -- same flag, same fields.",
    )
    args = ap.parse_args()

    records = load_antshield_dataset(args.dataset_root, include_metadata=args.include_metadata_fields)
    split_fn = (
        (lambda rid: assign_split_campaign(rid, args.val_frac))
        if args.split_strategy == "campaign"
        else (lambda rid: assign_split(rid, args.train_frac, args.val_frac))
    )

    splits = {"train": [], "val": [], "test": []}
    total_dropped = 0
    total_windows_with_entities = 0

    for r in records:
        text, char_entities = build_char_entities(r.text, args.mask_recon_tokens, seed_prefix=r.record_id)
        windows, dropped = chunk_and_tokenize(
            text, char_entities, args.max_tokens, args.overlap_tokens, args.max_windows_per_record
        )
        total_dropped += dropped

        split_name = split_fn(r.record_id)

        for w in windows:
            if w["ner"]:
                total_windows_with_entities += 1
            splits[split_name].append(w)

    for split_name, examples in splits.items():
        # deliberately NOT named "*_gliner.jsonl" -- that pattern is used by
        # convert_to_gliner_format.py for the eval format (char-offset
        # {"text":..., "entities":...}), which is a different, incompatible
        # schema from this file's token-indexed {"tokenized_text":...,
        # "ner":...}. Same output dir, different names, so they can't
        # silently overwrite or be confused for each other.
        out_path = Path(args.output_dir) / f"{split_name}_finetune.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        n_with_entities = sum(1 for ex in examples if ex["ner"])
        print(f"{split_name}: {len(examples)} windows ({n_with_entities} with entities) -> {out_path}")

    print(f"\nTotal windows with entities: {total_windows_with_entities}")
    print(f"Entities dropped (misaligned or straddling a window edge): {total_dropped}")


if __name__ == "__main__":
    main()
