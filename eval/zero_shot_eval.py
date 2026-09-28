"""
Phase 0 baseline: run a pretrained GLiNER checkpoint zero-shot on the
converted AntShield test set, no fine-tuning. Reports per-type and overall
precision/recall/F1 using exact span+label match.

This tells you whether fine-tuning (Phase 1) is actually necessary before
investing in it.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from gliner import GLiNER

# resolve ../data relative to this file, not the caller's cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "data"))
from pii_type_mapping import all_gliner_labels  # noqa: E402
from chunking import predict_with_chunking  # noqa: E402


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def spans_match(pred, gold):
    return pred["start"] == gold["start"] and pred["end"] == gold["end"] and pred["label"] == gold["label"]


def evaluate(model, examples, labels, threshold=0.5):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for ex in examples:
        text = ex["text"]
        gold_entities = ex["entities"]

        if not text.strip():
            continue

        predicted = predict_with_chunking(model, text, labels, threshold=threshold)
        pred_entities = [
            {"start": p["start"], "end": p["end"], "label": p["label"]} for p in predicted
        ]

        matched_gold = set()
        for p in pred_entities:
            hit = False
            for i, g in enumerate(gold_entities):
                if i in matched_gold:
                    continue
                if spans_match(p, g):
                    tp[p["label"]] += 1
                    matched_gold.add(i)
                    hit = True
                    break
            if not hit:
                fp[p["label"]] += 1

        for i, g in enumerate(gold_entities):
            if i not in matched_gold:
                fn[g["label"]] += 1

    print(f"\n{'label':<25}{'P':>8}{'R':>8}{'F1':>8}{'support':>10}")
    all_labels = sorted(set(tp) | set(fp) | set(fn))
    macro_f1 = []
    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        support = tp[label] + fn[label]
        macro_f1.append(f1)
        print(f"{label:<25}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{support:>10}")

    if macro_f1:
        print(f"\nMacro F1: {sum(macro_f1)/len(macro_f1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="converted JSONL test file")
    ap.add_argument("--model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model = GLiNER.from_pretrained(args.model)

    examples = load_jsonl(args.data)
    labels = all_gliner_labels()
    print(f"Evaluating on {len(examples)} examples, {len(labels)} candidate labels")

    evaluate(model, examples, labels, threshold=args.threshold)


if __name__ == "__main__":
    main()
