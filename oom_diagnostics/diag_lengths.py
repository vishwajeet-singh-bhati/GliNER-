#!/usr/bin/env python3
"""
READ-ONLY: GLiNER-style subword length distribution of fine-tuning JSONL files.

GLiNER feeds the encoder:  [prompt: <<ENT>> label ... <<SEP>>] + subwords of every word.
So the encoder length S is NOT the word count: one long word (base64 blob, hex string, URL)
becomes hundreds of subwords. Attention memory grows with B * heads * S^2.

This script reports, per file: word / subword percentiles, how many windows exceed
several S thresholds, the longest windows (lengths only, no text printed), how many
ENTITY spans live in windows above each threshold (= what dropping them would cost),
the batch-level attention-tensor size, and the S at which one attention tensor equals
the allocation size from your OOM message.

Usage (from repo root; no GPU, no torch needed, only the `tokenizers` package):
  python diagnostics/diag_lengths.py --tokenizer models/antshield-gliner-v1/tokenizer.json ^
         --data data/train_finetune.jsonl data/test_finetune.jsonl
Subword counts are estimated by encoding each word separately (what is_split_into_words does
for SentencePiece-style tokenizers); expect small deviations from the real run.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))


def pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))] if sorted_vals else 0


def prompt_len(tok, labels):
    return sum(1 + len(tok.encode(l, add_special_tokens=False).ids) for l in labels) + 1


def analyse(path, tok, p_len, args):
    rows = []  # (idx, n_words, S, max_word_sub, longest_word_chars, [labels])
    markers = 0
    batch, meta = [], []

    def flush():
        flat = [w for _, ws, _ in meta for w in ws]
        encs = tok.encode_batch(flat, add_special_tokens=False) if flat else []
        pos = 0
        for idx, ws, ner in meta:
            lens = [len(e.ids) for e in encs[pos:pos + len(ws)]]
            pos += len(ws)
            lw = max(range(len(ws)), key=lambda i: lens[i]) if ws else 0
            rows.append((idx, len(ws), sum(lens) + p_len + 2, max(lens) if lens else 0,
                         len(ws[lw]) if ws else 0, [n[2] for n in ner]))
        meta.clear()

    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            ws = ex["tokenized_text"]
            markers += sum(w in ("<<ENT>>", "<<SEP>>", "[SEP]", "[CLS]") for w in ws)
            meta.append((idx, ws, ex.get("ner", [])))
            if len(meta) >= 256:
                flush()
    flush()

    n = len(rows)
    S = sorted(r[2] for r in rows)
    Wd = sorted(r[1] for r in rows)
    print(f"\n==== {path}  ({n} windows) ====")
    print(f"prompt overhead for {len(args.labels)} labels: {p_len} subwords;  literal control tokens inside text: {markers}")
    print(f"words     p50={pct(Wd,.5)} p90={pct(Wd,.9)} p99={pct(Wd,.99)} max={Wd[-1]}")
    print(f"subwords  p50={pct(S,.5)} p90={pct(S,.9)} p99={pct(S,.99)} max={S[-1]}   mean inflation S/words = "
          f"{sum(r[2] for r in rows) / max(1, sum(r[1] for r in rows)):.2f}")
    print("windows above S threshold / entity spans inside them:")
    for T in args.thresholds:
        big = [r for r in rows if r[2] > T]
        print(f"  S>{T:<5} windows {len(big):>6} ({100*len(big)/n:5.2f}%)   entity spans {sum(len(r[5]) for r in big):>6}"
              f"   by label {dict(Counter(l for r in big for l in r[5]).most_common(6))}")
    print(f"top {args.top} longest windows (idx, words, subwords, max subwords in ONE word, chars of that word):")
    for r in sorted(rows, key=lambda r: -r[2])[:args.top]:
        print(f"  idx={r[0]:<7} words={r[1]:<4} S={r[2]:<6} max_sub_per_word={r[3]:<6} longest_word_chars={r[4]:<6} entities={len(r[5])}")

    B, H, L = args.batch_size, args.heads, args.layers
    print(f"\nattention-probability tensor [B={B}, H={H}, S, S]; padded to the longest sample in the batch:")
    for k in (1, 2, 3):
        s = S[-k] if len(S) >= k else 0
        print(f"  S={s:<6} bf16/layer {B*H*s*s*2/2**30:7.2f} GiB   fp32/layer {B*H*s*s*4/2**30:7.2f} GiB   "
              f"(x{L} layers kept for backward, bf16: {L*B*H*s*s*2/2**30:7.1f} GiB)")
    target = args.oom_gib * 2**30
    for name, nb in (("bf16", 2), ("fp32", 4)):
        s_star = int(math.sqrt(target / (B * H * nb)))
        p = sum(1 for r in rows if r[2] >= s_star) / n
        per_micro = 1 - (1 - p) ** B
        per_step = 1 - (1 - p) ** (B * args.grad_accum)
        print(f"  one [B,H,S,S] {name} tensor = {args.oom_gib} GiB  <=> S ~ {s_star}; windows with S>={s_star}: "
              f"{p*100:.3f}%  P(hit per micro-batch)={per_micro:.4f}  P(hit per optimizer step)={per_step:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the model being fine-tuned")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--thresholds", nargs="+", type=int, default=[384, 512, 768, 1024, 2048, 4096])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--heads", type=int, default=12, help="encoder attention heads (mdeberta-v3-base = 12)")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--oom-gib", type=float, default=2.97, help="size of the failed allocation")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    from pii_type_mapping import all_gliner_labels
    tok = Tokenizer.from_file(args.tokenizer)
    args.labels = all_gliner_labels()
    p_len = prompt_len(tok, args.labels)
    for p in args.data:
        analyse(p, tok, p_len, args)


if __name__ == "__main__":
    main()