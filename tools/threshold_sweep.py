"""
threshold_sweep.py - compare models on the SAME windows. Read-only: never trains
and never writes checkpoints.

Each model spec is one of
  <ckpt dir>      QAT checkpoint, run QUANTIZED (loads pytorch_model.bin)
  fpw:<ckpt dir>  the SAME checkpoint's latent weights run UNQUANTIZED
                  (isolates the cost of quantization for those exact weights)
  fp:<model dir>  a separate unquantized model, e.g. the 1-epoch bf16 base

predict_entities runs ONCE per window at a low score floor; P/R/F1 for every
threshold are derived from the stored scores (GLiNER decoding is greedy by
score, so higher-threshold spans are a subset of lower-threshold ones).
Windows that fail (e.g. CUDA OOM) are excluded for ALL models so the comparison
stays fair, and their count is printed.

Quick check : --n 1500 --include-negatives
Full test   : --n 100000 --include-negatives   (all windows; slow, prints progress)
"""
import argparse
import contextlib
import gc
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "quantization"))

import torch  # noqa: E402
from finetune_gliner import (load_jsonl, sanitize_example_in_place,  # noqa: E402
                             smart_load_state_dict, quantize_gliner_encoder)
from length_utils import fit_dataset  # noqa: E402
from gliner import GLiNER  # noqa: E402

THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
MIN_GOLD = 30     # per-label results with fewer gold spans are flagged unreliable


def gold_spans(ex):
    toks = ex["tokenized_text"]
    starts, ends, cur = [], [], 0
    for i, t in enumerate(toks):
        if i > 0:
            cur += 1
        starts.append(cur)
        cur += len(t)
        ends.append(cur)
    return {(starts[s], ends[e], str(l)) for s, e, l in (ex.get("ner") or [])
            if 0 <= s <= e < len(toks)}


def predict_all(model, data, labels, floor, use_autocast):
    """List aligned with data; None where prediction failed."""
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if use_autocast and torch.cuda.is_available() else contextlib.nullcontext())
    out = []
    with ctx:
        for i, ex in enumerate(data):
            if i and i % 2000 == 0:
                print(f"    ... {i}/{len(data)}", flush=True)
            text = " ".join(ex["tokenized_text"])
            try:
                preds = model.predict_entities(text, labels, threshold=floor)
                out.append([(int(p["start"]), int(p["end"]), str(p["label"]), float(p["score"]))
                            for p in preds])
            except Exception:
                out.append(None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return out


def score_at(preds, golds, thr):
    """{label: [tp, fp, fn]} plus overall under '__all__'."""
    c = defaultdict(lambda: [0, 0, 0])
    for pr, g in zip(preds, golds):
        p = {(s, e, l) for s, e, l, sc in pr if sc >= thr}
        for _, _, l in p & g:
            c[l][0] += 1
        for _, _, l in p - g:
            c[l][1] += 1
        for _, _, l in g - p:
            c[l][2] += 1
    c["__all__"] = [sum(v[i] for v in c.values()) for i in range(3)]
    return c


def prf(tp, fp, fn):
    p = 100 * tp / (tp + fp) if tp + fp else 0.0
    r = 100 * tp / (tp + fn) if tp + fn else 0.0
    f = 100 * 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return p, r, f


def to_device(model):
    if not torch.cuda.is_available():
        return
    try:
        model.to("cuda")
    except Exception:
        (model.model if hasattr(model, "model") else model).to("cuda")


def load_ckpt_weights(model, ckpt_dir):
    sd = torch.load(str(Path(ckpt_dir) / "pytorch_model.bin"), map_location="cpu")
    matched, total, missing, unexpected = smart_load_state_dict(model, sd)
    print(f"  restored {matched}/{total} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    del sd


def display_name(spec):
    if spec.startswith("fpw:"):
        return "FPW:" + Path(spec[4:]).name
    if spec.startswith("fp:"):
        return "FP:" + Path(spec[3:]).name
    return Path(spec).name


def load_model(base, spec, scaling_mode, max_types):
    if spec.startswith("fp:"):
        model = GLiNER.from_pretrained(spec[3:], local_files_only=True)
        model.config.max_types = max_types
    elif spec.startswith("fpw:"):
        model = GLiNER.from_pretrained(base, local_files_only=True)
        model.config.max_types = max_types
        load_ckpt_weights(model, spec[4:])          # latent weights, NO quantize call
    else:
        model = GLiNER.from_pretrained(base, local_files_only=True)
        model.config.max_types = max_types
        quantize_gliner_encoder(model, scaling_mode=scaling_mode, learnable=True)
        load_ckpt_weights(model, spec)
    to_device(model)
    (model.model if hasattr(model, "model") else model).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="dir the QAT checkpoints were trained from")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--sample", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-negatives", action="store_true",
                    help="also use windows with NO entities (needed to see the real false-positive rate)")
    ap.add_argument("--budget", type=int, default=0, help="trim windows to N text subwords (0 = off)")
    ap.add_argument("--table-thr", type=float, default=0.5, help="threshold for the per-label table")
    ap.add_argument("--scaling-mode", default="row_sigma_learnable")
    ap.add_argument("--max-types", type=int, default=17)
    ap.add_argument("--autocast", action="store_true")
    args = ap.parse_args()

    raw_train = load_jsonl(Path(args.data_dir) / "train_finetune.jsonl")
    labels = sorted({str(s[2]) for ex in raw_train for s in (ex.get("ner") or []) if len(s) >= 3})
    del raw_train

    pool = load_jsonl(Path(args.data_dir) / "test_finetune.jsonl")
    if not args.include_negatives:
        pool = [ex for ex in pool if ex.get("ner")]
    if args.sample == "random":
        random.Random(args.seed).shuffle(pool)
    val = [sanitize_example_in_place(ex) for ex in pool[:args.n]]
    n_pos = sum(1 for ex in val if ex.get("ner"))
    print(f"{len(val)} windows ({n_pos} with entities, {len(val) - n_pos} without), {len(labels)} labels")

    names, results, golds = [], {}, None
    for spec in args.models:
        name = display_name(spec)
        names.append(name)
        print(f"\n--- predicting with {name} ---", flush=True)
        model = load_model(args.base_model, spec, args.scaling_mode, args.max_types)
        if golds is None:
            tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
            if args.budget > 0 and tok is not None:
                val, _ = fit_dataset(val, tok, args.budget)
            golds = [gold_spans(ex) for ex in val]
        results[name] = predict_all(model, val, labels, floor=0.05, use_autocast=args.autocast)
        print(f"  failed windows: {sum(p is None for p in results[name])}", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = [i for i in range(len(val)) if all(results[n][i] is not None for n in names)]
    print(f"\nscoring on {len(ok)} windows that every model handled")
    g_ok = [golds[i] for i in ok]
    preds_ok = {n: [results[n][i] for i in ok] for n in names}
    ts = sorted(set(THRESHOLDS) | {args.table_thr})
    table = {n: {t: score_at(preds_ok[n], g_ok, t) for t in ts} for n in names}

    for n in names:
        print(f"\n=== {n} ===")
        print(f"  {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}")
        best = (-1, 0)
        for t in ts:
            p, r, f = prf(*table[n][t]["__all__"])
            best = max(best, (f, t))
            print(f"  {t:5.2f} {p:6.2f} {r:6.2f} {f:6.2f}")
        print(f"  best F1 {best[0]:.2f} at threshold {best[1]}")

    gold_n = {l: sum(1 for g in g_ok for _, _, ll in g if ll == l) for l in labels}
    w = 18
    print(f"\n=== per-label at threshold {args.table_thr} (P/R) ===  ('*' = fewer than {MIN_GOLD} gold spans: unreliable)")
    print(f"{'label':24s} {'gold':>6} " + " ".join(f"{n[-w:]:>{w}}" for n in names))
    for l in labels:
        cells = []
        for n in names:
            p, r, _ = prf(*table[n][args.table_thr].get(l, [0, 0, 0]))
            cells.append(f"{p:6.1f}/{r:5.1f}")
        flag = "*" if gold_n[l] < MIN_GOLD else " "
        print(f"{l:24s} {gold_n[l]:6d}{flag}" + " ".join(f"{c:>{w}}" for c in cells))

    print(f"\n=== per-label BEST threshold (thr / F1), from the grid {THRESHOLDS} ===")
    print(f"{'label':24s} {'gold':>6} " + " ".join(f"{n[-w:]:>{w}}" for n in names))
    for l in labels:
        cells = []
        for n in names:
            if gold_n[l] == 0:
                cells.append("-")
                continue
            best = max(((prf(*table[n][t].get(l, [0, 0, 0]))[2], t) for t in ts))
            cells.append(f"{best[1]:.2f} / {best[0]:5.1f}")
        flag = "*" if gold_n[l] < MIN_GOLD else " "
        print(f"{l:24s} {gold_n[l]:6d}{flag}" + " ".join(f"{c:>{w}}" for c in cells))


if __name__ == "__main__":
    main()