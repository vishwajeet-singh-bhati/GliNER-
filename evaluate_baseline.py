import json
import time
from pprint import pprint
from collections import Counter
from gliner import GLiNER

# ── Load validation data ──
val_data = []
with open('data/val_finetune.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        val_data.append(json.loads(line))

# ── Check for RECON tokens in val set ──
recon_count = 0
for w in val_data[:200]:
    text = " ".join(w.get("tokenized_text", []))
    if "RECON_" in text:
        recon_count += 1
if recon_count > 0:
    print(f"⚠️  WARNING: {recon_count}/200 sampled val windows contain RECON_ tokens!")
    print("   The val set has NOT been synthesized. Evaluation will be unreliable.")
    print("   Run the RECON replacement pipeline on val_finetune.jsonl first.\n")
else:
    print("✅ No RECON_ tokens found in the first 200 validation windows.")

# ── All PII labels used in your dataset ──
ALL_LABELS = [
    "advertiser id", "android device id", "location coordinates",
    "city", "imei number", "username", "zip code",
    "device serial number", "first name", "imsi number",
    "gender", "email address", "password", "phone number",
    "last name", "sim card iccid"
]

models_to_test = {
    "base (zero-shot)": "urchade/gliner_multi_pii-v1",
}

def per_class_eval(model, val_data, labels, threshold=0.5):
    """Compute per-class P/R/F1 with span-level exact match."""
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for i, window in enumerate(val_data):
        text = " ".join(window.get("tokenized_text", []))
        if not text.strip():
            continue

        # Gold spans: set of (start_tok, end_tok, label)
        gold = set()
        for span in window.get("ner", []):
            gold.add((span[0], span[1], span[2]))

        # Predicted spans
        try:
            preds = model.predict_entities(text, labels, threshold=threshold)
        except Exception:
            preds = []

        # Convert predictions to token-level spans for matching
        pred_set = set()
        for p in preds:
            pred_set.add((p["text"].strip(), p["label"]))

        gold_text_set = set()
        for s in window.get("ner", []):
            tok_text = " ".join(window["tokenized_text"][s[0]:s[1]+1])
            gold_text_set.add((tok_text.strip(), s[2]))

        for item in pred_set & gold_text_set:
            tp[item[1]] += 1
        for item in pred_set - gold_text_set:
            fp[item[1]] += 1
        for item in gold_text_set - pred_set:
            fn[item[1]] += 1

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(val_data)} windows...")

    # Print per-class results
    print(f"\n{'Label':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 75)
    total_tp = total_fp = total_fn = 0
    for label in sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))):
        t, f, n = tp[label], fp[label], fn[label]
        total_tp += t; total_fp += f; total_fn += n
        p = t / (t + f) if (t + f) > 0 else 0
        r = t / (t + n) if (t + n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{label:<25} {t:>5} {f:>5} {n:>5} {p:>6.1%} {r:>6.1%} {f1:>6.1%}")

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("-" * 75)
    print(f"{'MICRO TOTAL':<25} {total_tp:>5} {total_fp:>5} {total_fn:>5} {p:>6.1%} {r:>6.1%} {f1:>6.1%}")
    return {"precision": p, "recall": r, "f1": f1}

for name, path in models_to_test.items():
    print(f"\n{'='*70}")
    print(f"Model: {name}")
    print(f"Path:  {path}")
    print(f"{'='*70}")
    try:
        model = GLiNER.from_pretrained(path)
        results = per_class_eval(model, val_data, ALL_LABELS)
        # Also run GLiNER's built-in eval for comparison
        print("\n--- GLiNER built-in eval ---")
        builtin = model.evaluate(val_data, flat_ner=True)
        if isinstance(builtin, tuple):
            print(builtin[0])
        else:
            print(builtin)
        del model
    except Exception as e:
        print(f"  FAILED: {e}")
