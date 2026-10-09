"""
Phase 0 baseline: run a pretrained GLiNER checkpoint zero-shot on the
converted AntShield test set, no fine-tuning. Reports per-type and overall
precision/recall/F1 using exact span+label match.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
import torch

from gliner import GLiNER

# resolve ../data relative to this file, not the caller working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "data"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "quantization"))
from pii_type_mapping import all_gliner_labels  # noqa: E402
from chunking import predict_with_chunking  # noqa: E402
from quantize_gliner import load_quantized_gliner  # noqa: E402


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def spans_match(pred, gold):
    return pred["start"] == gold["start"] and pred["end"] == gold["end"] and pred["label"] == gold["label"]


def evaluate(model, examples, labels, threshold=0.5, verbose=True):
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

    if verbose:
        header = "{:<25}{:>8}{:>8}{:>8}{:>10}".format("label", "P", "R", "F1", "support")
        print(f"\n{header}")
        
    all_labels = sorted(set(tp) | set(fp) | set(fn))
    macro_f1 = []
    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        support = tp[label] + fn[label]
        macro_f1.append(f1)
        if verbose:
            print(f"{label:<25}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{support:>10}")

    overall_macro_f1 = sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0
    if verbose:
        print(f"\nMacro F1: {overall_macro_f1:.3f}")
    return overall_macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="converted JSONL test file")
    ap.add_argument("--model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--quantized", action="store_true",
        help="load --model as a quantized checkpoint via load_quantized_gliner() "
             "instead of plain GLiNER.from_pretrained() -- REQUIRED for any "
             "checkpoint trained with --quantize, since those have E8QATLinear "
             "layers that plain from_pretrained() cannot reconstruct correctly.",
    )
    ap.add_argument(
        "--base-model", default="urchade/gliner_multi_pii-v1",
        help="--quantized only: the original pretrained model to reconstruct "
             "the quantized architecture from before loading the checkpoint weights.",
    )
    ap.add_argument("--scaling-mode", default="block_max",
                     choices=["block_max", "row_sigma", "row_sigma_learnable"])
    ap.add_argument("--no-learnable-scaling", dest="learnable_scaling", action="store_false")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    if args.quantized:
        model = load_quantized_gliner(
            args.base_model, args.model,
            scaling_mode=args.scaling_mode, learnable=args.learnable_scaling)
    else:
        model = GLiNER.from_pretrained(args.model)

    if torch.cuda.is_available():
        model = model.to("cuda")

    examples = load_jsonl(args.data)
    labels = all_gliner_labels()
    print(f"Evaluating on {len(examples)} examples, {len(labels)} candidate labels")

    with torch.no_grad():
        evaluate(model, examples, labels, threshold=args.threshold)


if __name__ == "__main__":
    main()