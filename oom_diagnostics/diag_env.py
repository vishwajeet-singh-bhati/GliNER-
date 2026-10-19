#!/usr/bin/env python3
"""
READ-ONLY environment / script / GLiNER-source / checkpoint diagnostics for the OOM investigation.
Imports torch only if installed; never loads a model, never writes a file.

  python diagnostics/diag_env.py --script train/finetune_gliner.py --models-dir models [--deep]
--deep additionally loads each checkpoint's weights on CPU and checks every tensor is finite.
"""
import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


def sec(t):
    print(f"\n==== {t} ====")


def versions():
    sec("1. Versions / GPU")
    print(sys.version.replace("\n", " "), "|", platform.platform())
    for p in ("torch", "gliner", "transformers", "tokenizers", "accelerate", "safetensors", "sentencepiece", "numpy"):
        try:
            print(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            print(f"{p}: NOT INSTALLED")
    print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"))
    try:
        import torch
        print("torch.version.cuda:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            print(f"GPU0 {pr.name}: total {total/2**30:.2f} GiB, free right now {free/2**30:.2f} GiB, bf16 supported: {torch.cuda.is_bf16_supported()}")
    except Exception as e:  # noqa: BLE001
        print("torch check skipped:", e)
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"],
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]):
        try:
            print(subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or "(no output)")
        except Exception as e:  # noqa: BLE001
            print("nvidia-smi unavailable:", e)


KEYS = r"max_len|max_length|truncat|gradient_checkpoint|resume|bf16|fp16|group_by_length|dataloader|num_workers|empty_cache|OutOfMemory|except|max_types|max_width|save_steps|save_total_limit|eval_|per_device|shuffle|sort"


def scan_script(path):
    sec(f"2. Training script: {path}")
    p = Path(path)
    if not p.exists():
        print("MISSING (restore from git with: git show HEAD:train/finetune_gliner.py)")
        return
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"{len(lines)} lines, modified {time.ctime(p.stat().st_mtime)}")
    for i, l in enumerate(lines, 1):
        if "add_argument" in l or re.search(KEYS, l, re.I):
            print(f"{i:4}: {l.strip()[:150]}")


def gliner_source():
    sec("3. GLiNER source (grep only)")
    spec = importlib.util.find_spec("gliner")
    if not spec or not spec.origin:
        print("gliner not importable from this interpreter")
        return
    root = Path(spec.origin).parent
    print("package dir:", root)
    for f in sorted(root.rglob("*.py")):
        src = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, l in enumerate(src):
            if re.search(r"def (extract_prompt_features\w*|train_model|create_dataloader|collate_fn)\b", l):
                print(f"\n{f.relative_to(root)}:{i+1}: {l.strip()}")
                indent = len(l) - len(l.lstrip())
                for j in range(i + 1, min(i + 80, len(src))):
                    s = src[j]
                    if s.strip() and len(s) - len(s.lstrip()) <= indent and s.lstrip().startswith(("def ", "class ")):
                        break
                    if re.search(r"arange|expand|zeros|\.max\(|max_len|truncat|gradient_checkpoint|empty_cache", s):
                        print(f"    {j+1}: {s.strip()[:140]}")
    print("\n-- length / truncation handling anywhere in gliner:")
    n = 0
    for f in sorted(root.rglob("*.py")):
        for i, l in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if re.search(r"max_len\b|truncation|max_length", l) and n < 60:
                print(f"{f.relative_to(root)}:{i}: {l.strip()[:140]}")
                n += 1


def checkpoints(models_dir, deep):
    sec(f"4. Checkpoints under {models_dir}")
    base = Path(models_dir)
    for d in sorted(list(base.glob("*/checkpoint-*")) + list(base.glob("checkpoint-*"))):
        files = {f.name: f.stat().st_size for f in d.iterdir() if f.is_file()}
        state = d / "trainer_state.json"
        info = ""
        if state.exists():
            try:
                ts = json.loads(state.read_text(encoding="utf-8"))
                last = ts.get("log_history", [{}])[-1]
                info = f"global_step={ts.get('global_step')} max_steps={ts.get('max_steps')} last_log={ {k: last[k] for k in ('loss', 'learning_rate', 'step') if k in last} }"
            except Exception as e:  # noqa: BLE001
                info = f"trainer_state unreadable: {e}"
        weights = [n for n in files if n in ("pytorch_model.bin", "model.safetensors")]
        resumable = state.exists() and "optimizer.pt" in files and "scheduler.pt" in files and bool(weights)
        print(f"\n{d}  [{time.ctime(d.stat().st_mtime)}]  total {sum(files.values())/2**30:.2f} GiB")
        print(f"  files: {sorted(files)}")
        print(f"  {info}\n  weights: {weights}  full-resume files present (weights+optimizer+scheduler+trainer_state): {resumable}")
        if deep and weights:
            try:
                import torch
                w = d / weights[0]
                sd = (__import__('safetensors.torch', fromlist=['load_file']).load_file(str(w)) if w.suffix == ".safetensors"
                      else torch.load(w, map_location="cpu", weights_only=True))
                bad = [k for k, v in sd.items() if hasattr(v, "is_floating_point") and v.is_floating_point() and not bool(v.isfinite().all())]
                print(f"  deep check: {len(sd)} tensors, non-finite tensors: {len(bad)} {bad[:3]}")
            except Exception as e:  # noqa: BLE001
                print("  deep check failed:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", default="train/finetune_gliner.py")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()
    versions()
    scan_script(a.script)
    gliner_source()
    checkpoints(a.models_dir, a.deep)


if __name__ == "__main__":
    main()