"""
Evaluates every checkpoint-N directory under --checkpoints-dir against the
held-out test set and ranks them by macro F1.
"""
import argparse
import glob
import json
import os
import re
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "data"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "quantization"))

from pii_type_mapping import all_gliner_labels  # noqa: E402
from quantize_gliner import load_quantized_gliner  # noqa: E402
from zero_shot_eval import evaluate, load_jsonl  # noqa: E402


def find_checkpoints(checkpoints_dir):
    paths = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    with_step = []
    for p in paths:
        m = re.search(r"checkpoint-(\d+)$", p)
        if m:
            with_step.append((int(m.group(1)), p))
    return sorted(with_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", required=True)
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument(
        "--scaling-mode",
        default="block_max",
        choices=["block_max", "row_sigma", "row_sigma_learnable"],
    )
    ap.add_argument(
        "--no-learnable-scaling", dest="learnable_scaling", action="store_false"
    )
    ap.add_argument("--data", required=True, help="converted JSONL test file")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = full test set. For a quick ranking sweep across many "
        "checkpoints, set this to e.g. 1500 first, then re-run with 0 "
        "(full set) on just the top candidates.",
    )
    args = ap.parse_args()

    checkpoints = find_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        print(f"No checkpoint-N directories found under {args.checkpoints_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints: steps {[s for s, _ in checkpoints]}")

    examples = load_jsonl(args.data)
    if args.max_samples and args.max_samples < len(examples):
        examples = examples[: args.max_samples]
        print(
            f"Using first {len(examples)} test examples for this sweep "
            f"(pass --max-samples 0 for the full set once you have a shortlist)"
        )
    else:
        print(f"Using all {len(examples)} test examples")

    labels = all_gliner_labels()
    results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating using device: {device}")

    for step, path in checkpoints:
        print(f"\n--- checkpoint-{step} ---")
        try:
            model = load_quantized_gliner(
                args.base_model,
                path,
                scaling_mode=args.scaling_mode,
                learnable=args.learnable_scaling,
            )
            model.eval()
            if device == "cuda":
                model = model.to("cuda")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            results.append((step, None))
            continue

        with torch.no_grad():
            macro_f1 = evaluate(
                model, examples, labels, threshold=args.threshold, verbose=False
            )
        print(f"  macro F1: {macro_f1:.4f}")
        results.append((step, macro_f1))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    sep = "=" * 40
    print(f"\n{sep}\nSummary, ranked best to worst:\n{sep}")
    valid = [(s, f) for s, f in results if f is not None]
    valid.sort(key=lambda x: x[1], reverse=True)
    for step, f1 in valid:
        print(f"  checkpoint-{step:<8} macro F1: {f1:.4f}")
    failed = [s for s, f in results if f is None]
    if failed:
        print(f"\n  Failed to load: {failed}")


if __name__ == "__main__":
    main()