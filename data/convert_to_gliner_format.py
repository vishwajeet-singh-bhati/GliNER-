"""
Converts AntShield packet records into GLiNER's expected format:

    {"text": "...", "entities": [{"start": 12, "end": 24, "label": "email address"}]}

Span-finding strategy (confirmed against the real dataset, not guessed):
every leaked PII value in this public release is redacted with a literal
token "RECON_<Type>" (e.g. "RECON_AndroidId"). We regex-match those tokens
directly -- no string-matching against unknown real values needed, and no
regex/heuristic guessing either. Verified: 8579/8579 (100%) of positive
records carry at least one such token in the extracted text.

--mask-recon-tokens (recommended, off by default):
  Replaces each "RECON_<Type>" token with a synthetic realistic-looking value
  of the correct type/shape (data/recon_synthesizer.py) before span offsets
  are finalized. Without this flag, the entity span's literal text is
  "RECON_AndroidId" etc. -- which spells out its own label, so a model can
  solve the task by substring-matching the token instead of learning any real
  PII pattern. Turn this on for a result that means anything about actual
  detection capability; leave it off only if you specifically want a
  literal-replay comparison against how the binary BERT model was trained
  (which likely has the same leakage problem via the bare "RECON_" prefix).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_type_mapping import ANTSHIELD_TO_GLINER_LABEL, RAW_TYPES_BY_LENGTH  # noqa: E402
from antshield_loader import load_antshield_dataset, PacketRecord  # noqa: E402
from recon_synthesizer import synthesize  # noqa: E402
from splitting import assign_split  # noqa: E402

# alternation ordered longest-first so e.g. "AndroidId" can't be cut short by
# a shorter type name that happens to be a prefix; also correctly separates
# adjacent tokens with no delimiter, e.g. "RECON_SerialNumberRECON_AndroidId"
_RECON_PATTERN = re.compile("RECON_(" + "|".join(RAW_TYPES_BY_LENGTH) + ")")


def find_recon_spans(text: str):
    """Yields (start, end, raw_type) for every RECON_<Type> token in text, in order."""
    for m in _RECON_PATTERN.finditer(text):
        yield m.start(), m.end(), m.group(1)


def convert_record(record: PacketRecord, mask_recon_tokens: bool):
    text = record.text
    matches = list(find_recon_spans(text))

    if not mask_recon_tokens:
        entities = [
            {"start": s, "end": e, "label": ANTSHIELD_TO_GLINER_LABEL[raw_type]}
            for s, e, raw_type in matches
        ]
        return {"text": text, "entities": entities}

    # rebuild the string, replacing each token with a synthesized value and
    # tracking offsets as they shift
    out_parts = []
    entities = []
    cursor = 0
    for i, (s, e, raw_type) in enumerate(matches):
        out_parts.append(text[cursor:s])
        label = ANTSHIELD_TO_GLINER_LABEL[raw_type]
        replacement = synthesize(label, seed_key=f"{record.record_id}:{i}")
        new_start = sum(len(p) for p in out_parts)
        out_parts.append(replacement)
        new_end = new_start + len(replacement)
        entities.append({"start": new_start, "end": new_end, "label": label})
        cursor = e
    out_parts.append(text[cursor:])

    return {"text": "".join(out_parts), "entities": entities}


def convert(records, mask_recon_tokens: bool):
    examples = []
    n_pii_types_declared = 0
    n_recon_tokens_found = 0

    for r in records:
        n_pii_types_declared += len(r.pii_types)
        ex = convert_record(r, mask_recon_tokens)
        n_recon_tokens_found += len(ex["entities"])
        examples.append(ex)

    print(f"Converted {len(examples)} examples")
    print(f"pii_types declared across dataset: {n_pii_types_declared}")
    print(f"RECON_ tokens actually located and turned into spans: {n_recon_tokens_found}")
    if n_recon_tokens_found < n_pii_types_declared:
        print(
            f"[warn] {n_pii_types_declared - n_recon_tokens_found} declared pii_types had no "
            f"matching RECON_ token found in the extracted text fields -- "
            f"inspect data/antshield_loader.py's field list if this number is large."
        )

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="path to extracted antshield_public_dataset/")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument(
        "--mask-recon-tokens",
        action="store_true",
        help="replace RECON_<Type> tokens with synthetic realistic values (recommended)",
    )
    ap.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="filter to one split using the same record_id hashing as "
             "prepare_finetune_data.py, so this eval-format file and that "
             "script's *_finetune.jsonl agree on which records are 'test' "
             "(default 'all' -- fine for a pure zero-shot run with no "
             "fine-tuning to leak against, but use 'test' when evaluating a "
             "fine-tuned checkpoint)",
    )
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    records = load_antshield_dataset(args.dataset_root)
    if args.split != "all":
        records = [
            r for r in records
            if assign_split(r.record_id, args.train_frac, args.val_frac) == args.split
        ]
        print(f"Filtered to '{args.split}' split: {len(records)} records")

    examples = convert(records, mask_recon_tokens=args.mask_recon_tokens)

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
