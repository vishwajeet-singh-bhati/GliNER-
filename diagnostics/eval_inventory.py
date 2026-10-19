#!/usr/bin/env python3
"""
READ-ONLY inventory for the evaluation audit. Writes nothing, loads no model.
Establishes (1) which fine-tuned checkpoint exists and what it was trained with,
(2) which evaluation datasets exist and whether they are the ones build_dataset.py produced,
(3) train/test leakage counts and per-class REAL test support.
Prints counts and hashes only -- no traffic text.

  python diagnostics\\eval_inventory.py --model-dir D:\\models\\antshield-gliner-bf16-1epoch [--deep]
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "data"))


def sec(t):
    print(f"\n==== {t} ====")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def versions():
    sec("1. Versions / GPU")
    print(sys.version.split()[0])
    for p in ("torch", "gliner", "transformers", "tokenizers", "safetensors", "numpy"):
        try:
            print(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            print(f"{p}: NOT INSTALLED")
    try:
        import torch
        print("cuda available:", torch.cuda.is_available(),
              "| bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:  # noqa: BLE001
        print("torch check skipped:", e)


def model_dir(d, deep):
    sec(f"2. Fine-tuned model dir: {d}")
    d = Path(d)
    if not d.exists():
        print("MISSING")
        return
    for f in sorted(d.iterdir()):
        print(f"  {f.name:<32}{(f.stat().st_size / 2**20 if f.is_file() else 0):>10.1f} MiB  {time.ctime(f.stat().st_mtime)}")
    cfg = d / "gliner_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        print("gliner_config.json:", json.dumps({k: v for k, v in c.items() if not isinstance(v, (dict, list))}, indent=1)[:1500])
    else:
        print("no gliner_config.json  (GLiNER.from_pretrained would not find a config)")
    tok = d / "tokenizer.json"
    if tok.exists():
        print("tokenizer.json sha256:", sha256(tok)[:16])
        for ref in sorted((REPO / "models").glob("*/tokenizer.json"))[:3]:
            print(f"   same as {ref.parent.name}: {sha256(ref)[:16] == sha256(tok)[:16]}")
    states = sorted(list(d.glob("trainer_state.json")) + list(d.glob("checkpoint-*/trainer_state.json")),
                    key=lambda p: json.loads(p.read_text(encoding="utf-8")).get("global_step", 0))
    if states:
        ts = json.loads(states[-1].read_text(encoding="utf-8"))
        logs = [l for l in ts.get("log_history", []) if "loss" in l or "train_loss" in l]
        print(f"trainer_state ({states[-1].relative_to(d)}): global_step={ts.get('global_step')} max_steps={ts.get('max_steps')} "
              f"epoch={ts.get('epoch')} train_batch_size={ts.get('train_batch_size')}")
        print("   last loss entries:", [{k: l[k] for k in ('step', 'loss', 'train_loss') if k in l} for l in logs[-3:]])
    else:
        print("no trainer_state.json found (cannot verify steps/epoch/loss from artifacts)")
    w = next((d / n for n in ("model.safetensors", "pytorch_model.bin") if (d / n).exists()), None)
    print("weights file:", w.name if w else "NONE FOUND")
    if deep and w:
        try:
            import torch
            sd = (__import__("safetensors.torch", fromlist=["load_file"]).load_file(str(w)) if w.suffix == ".safetensors"
                  else torch.load(w, map_location="cpu", weights_only=True))
            dt = Counter(str(v.dtype) for v in sd.values() if hasattr(v, "dtype"))
            bad = sum(1 for v in sd.values() if hasattr(v, "is_floating_point") and v.is_floating_point() and not bool(v.isfinite().all()))
            print(f"   deep: {len(sd)} tensors, dtypes {dict(dt)}, non-finite tensors {bad}")
        except Exception as e:  # noqa: BLE001
            print("   deep check failed:", e)
    return states[-1] if states else None


def load(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                rows.append(json.loads(l))
    return rows


def spans_of(ex):
    return [(s[2]) for s in ex["ner"]] if "ner" in ex else [e["label"] for e in ex.get("entities", [])]


def datasets(data_dir):
    sec(f"3. Candidate evaluation / training files under {data_dir}")
    d = Path(data_dir)
    report = d / "build_report.json"
    recorded = json.loads(report.read_text(encoding="utf-8")).get("sha256", {}) if report.exists() else {}
    print("build_report.json:", "present" if report.exists() else "ABSENT (files were not made by build_dataset.py, or report moved)")
    data = {}
    for name in ("train_finetune.jsonl", "test_finetune.jsonl", "test_gliner.jsonl", "val_finetune.jsonl"):
        p = d / name
        if not p.exists():
            print(f"{name:<24} absent")
            continue
        rows, keysets, bad = [], Counter(), 0
        try:
            rows = load(p)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: parse error {e}")
            continue
        for r in rows:
            keysets[tuple(sorted(r))] += 1
        h = sha256(p)
        prov = "n/a" if name not in recorded else ("MATCHES build_report" if recorded[name] == h else "DIFFERS from build_report")
        print(f"{name:<24}{len(rows):>8} lines  {time.ctime(p.stat().st_mtime)}  sha256 {h[:16]}  provenance: {prov}")
        print(f"   schema keysets: {dict(keysets)}")
        data[name] = rows
        man = d / name.replace("_finetune", "_manifest")
        if man.exists():
            m = load(man)
            print(f"   manifest: {len(m)} lines (aligned: {len(m) == len(rows)}), sources {dict(Counter(x['source'] for x in m))}")
    return data


def leakage(data, d):
    sec("4. Train/test leakage (counts only)")
    if "train_finetune.jsonl" not in data or "test_finetune.jsonl" not in data:
        print("need both train_finetune.jsonl and test_finetune.jsonl; skipped")
        return
    tr, te = data["train_finetune.jsonl"], data["test_finetune.jsonl"]
    fp = lambda w: hashlib.md5(json.dumps([w["tokenized_text"], w["ner"]], separators=(",", ":")).encode()).hexdigest()
    ft = lambda w: hashlib.md5("\x1f".join(w["tokenized_text"]).encode()).hexdigest()
    nz = lambda w: re.sub(r"\d+", "0", " ".join(w["tokenized_text"]).lower())
    te_fp, te_ft, te_nz = {fp(w) for w in te}, {ft(w) for w in te}, {nz(w) for w in te}
    print(f"train windows identical (tokens+ner) to a test window : {sum(fp(w) in te_fp for w in tr)}")
    print(f"train windows with identical tokens (any labels)      : {sum(ft(w) in te_ft for w in tr)}")
    print(f"train windows equal after lowercasing + digits->0     : {sum(nz(w) in te_nz for w in tr)}")
    try:
        mt, ms = load(d / "train_manifest.jsonl"), load(d / "test_manifest.jsonl")
        ids_tr = {m["record_id"] for m in mt if m["source"] == "real"}
        ids_te = {m["record_id"] for m in ms if "record_id" in m}
        print(f"source flows present in both splits                   : {len(ids_tr & ids_te)}")
        print(f"non-real (synthetic/decoy) windows in TEST            : {sum(m['source'] != 'real' for m in ms)}")
        print(f"synthetic windows in TRAIN                            : {sum(m['source'] != 'real' for m in mt)}")
    except FileNotFoundError:
        print("manifests absent: source-flow overlap and real/synthetic split cannot be verified")
    c = Counter(l for w in te for l in spans_of(w))
    print("\nREAL test support (entity spans per label):")
    for lab, n in sorted(c.items(), key=lambda x: x[1]):
        print(f"  {lab:<28}{n:>7}   {'<-- too few for reliable per-class conclusions' if n < 30 else ''}")
    try:
        from pii_type_mapping import all_gliner_labels
        labs = all_gliner_labels()
        print(f"\nall_gliner_labels(): {len(labs)} labels; zero-support in test: {[l for l in labs if c[l] == 0]}")
    except Exception as e:  # noqa: BLE001
        print("could not import pii_type_mapping:", e)


def epochs(state_path, data):
    sec("5. 'One epoch' sanity check")
    if not state_path or "train_finetune.jsonl" not in data:
        print("need trainer_state.json and train_finetune.jsonl")
        return
    ts = json.loads(state_path.read_text(encoding="utf-8"))
    n = len(data["train_finetune.jsonl"])
    print(f"train windows now on disk: {n}. Samples seen = steps x batch x accum = {ts.get('global_step')} x "
          f"{ts.get('train_batch_size')} x <grad_accum from your command line (4)>.")
    print("If steps count optimizer updates: epochs ~", None if not ts.get("train_batch_size") else
          round(ts["global_step"] * ts["train_batch_size"] * 4 / n, 2),
          "(valid only if the file on disk is the file trained on -- see provenance above)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=r"D:\models\antshield-gliner-bf16-1epoch")
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--deep", action="store_true", help="load weights on CPU: dtypes + finite check")
    a = ap.parse_args()
    versions()
    st = model_dir(a.model_dir, a.deep)
    data = datasets(a.data_dir)
    leakage(data, Path(a.data_dir))
    epochs(st, data)


if __name__ == "__main__":
    main()