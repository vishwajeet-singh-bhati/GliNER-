"""
count_long.py - READ-ONLY. Counts how many training windows would be dropped at
several subword limits (nothing is dropped, nothing is written).

Uses the same cleanup as training first (256-word window, 128-char word clip),
so the numbers match what the training run would actually see.

  python tools/count_long.py --tokenizer D:/models/antshield-gliner-bf16-1epoch
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from finetune_gliner import load_jsonl, sanitize_example_in_place  # noqa: E402
from long_filter import count_subwords  # noqa: E402


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q / 100 * len(vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-window-tokens", type=int, default=256)
    ap.add_argument("--max-token-chars", type=int, default=128)
    ap.add_argument("--limits", type=int, nargs="+",
                    default=[512, 640, 768, 896, 1024, 1280, 1536])
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    raw = load_jsonl(Path(args.data_dir) / "train_finetune.jsonl")
    data = [sanitize_example_in_place(e, args.max_window_tokens, args.max_token_chars) for e in raw]
    counts = [count_subwords(tok, ex["tokenized_text"]) for ex in data]
    labels_per = [[str(s[2]) for s in (ex.get("ner") or []) if len(s) >= 3] for ex in data]

    total_span = Counter(l for ls in labels_per for l in ls)
    n = len(data)
    print(f"{n} training windows after the training script's own cleanup")
    print(f"subwords: p50={pct(counts,50)} p90={pct(counts,90)} p99={pct(counts,99)} max={max(counts)}\n")

    print(f"{'limit':>6} {'windows dropped':>16} {'% of train':>11} {'windows kept':>13}")
    for L in args.limits:
        d = sum(c > L for c in counts)
        print(f"{L:6d} {d:16d} {100*d/n:10.2f}% {n-d:13d}")

    print("\nentity spans lost, by label (count and % of that label's spans):")
    print(f"{'label':24s} {'total':>6} " + " ".join(f"{('>'+str(L)):>12}" for L in args.limits))
    for lab, tot in sorted(total_span.items(), key=lambda kv: -kv[1]):
        cells = []
        for L in args.limits:
            lost = sum(ls.count(lab) for ls, c in zip(labels_per, counts) if c > L)
            cells.append(f"{lost}({100*lost/tot:.1f}%)")
        print(f"{lab:24s} {tot:6d} " + " ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()