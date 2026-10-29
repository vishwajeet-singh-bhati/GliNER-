"""
Draw training curves from the console logs of finetune_gliner.py (and the FP checkpoint_leaderboard.csv).

Per log it plots (whatever the log contains):
    training loss vs step | running train F1 vs step | validation F1 / P / R at every epoch end |
    validation loss | cosine to the FP encoder ([qat-diag] lines, QAT stages)

Usage (PowerShell, repo root):
    python tools\\plot_training_curves.py --out curves `
        --log "stage1=D:\\gliner-antshield\\models\\clean\\stage1.log" `
        --log "stage2=D:\\gliner-antshield\\models\\clean\\stage2.log" `
        --log "stage3 (joint, 10 epochs)=D:\\gliner-antshield\\models\\clean\\stage3.log" `
        --fp-csv "D:\\gliner-antshield\\models\\fp-v2_6800steps_2026-10-24\\checkpoint_leaderboard.csv"
    python tools\\plot_training_curves.py --selftest

Writes one PNG per log (<out>\\<name>.png) plus <out>\\curves_data.csv with every parsed point.
Needs matplotlib (pip install matplotlib).
"""
import argparse
import csv
import re
import sys
import tempfile
from pathlib import Path

EPOCH_LINE = re.compile(r"Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] \(Global (\d+)/(\d+)\) Loss:\s*([\d.]+) Acc:\s*([\d.]+)% F1:\s*([\d.]+)%")
DIAG = re.compile(r"\[qat-diag\] step ~\s*(\d+) \| cos\(student,FP\) ([\d.]+)(?: \| cos CENTERED ([\d.]+))?")
VAL_LOSS = re.compile(r"Validation Loss\s*:\s*([\d.]+)")
VAL_LINE = re.compile(r"Validation Acc\s*:\s*([\d.]+)% \(Val F1: ([\d.]+)% \| P: ([\d.]+)% \| R: ([\d.]+)%")
BLOCK = re.compile(r"Epoch (\d+)/(\d+) \|[^\n]*\(Step (\d+)/(\d+)\)")


def parse(text):
    train = [(int(m[4]), float(m[6]), float(m[7]), float(m[8])) for m in EPOCH_LINE.findall(text)]
    diag = [(int(a), float(b), float(c) if c else None) for a, b, c in DIAG.findall(text)]
    val = []
    cur = None
    for line in text.splitlines():
        b = BLOCK.search(line)
        if b:
            cur = {"epoch": int(b.group(1)), "step": int(b.group(3))}
            val.append(cur)
            continue
        if cur is None:
            continue
        m = VAL_LOSS.search(line)
        if m:
            cur["val_loss"] = float(m.group(1))
        m = VAL_LINE.search(line)
        if m:
            cur.update(val_acc=float(m.group(1)), val_f1=float(m.group(2)), val_p=float(m.group(3)), val_r=float(m.group(4)))
    val = [v for v in val if "val_f1" in v]
    return train, diag, val


def plot_log(name, text, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train, diag, val = parse(text)
    panels = []
    if train:
        panels += ["loss", "f1"]
    if val and any("val_loss" in v for v in val):
        panels.append("val_loss")
    if diag:
        panels.append("cos")
    if not panels:
        print(f"  {name}: nothing to plot (no 'Epoch [..] Step [..]' lines found)")
        return train, diag, val
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    for ax, p in zip(axes[0], panels):
        ax.grid(True, color="#ececE8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("step")
        if p == "loss":
            ax.plot([t[0] for t in train], [t[1] for t in train], color="#2a78d6", lw=2)
            ax.set_title("training loss")
            if min(t[1] for t in train) > 0 and max(t[1] for t in train) / max(1e-9, min(t[1] for t in train)) > 30:
                ax.set_yscale("log")
        elif p == "f1":
            ax.plot([t[0] for t in train], [t[3] for t in train], color="#2a78d6", lw=2, label="train F1 (running)")
            if val:
                ax.plot([v["step"] for v in val], [v["val_f1"] for v in val], color="#eb6834", lw=2, marker="o", label="validation F1")
                ax.plot([v["step"] for v in val], [v["val_p"] for v in val], color="#1baf7a", lw=1.2, ls="--", label="val precision")
                ax.plot([v["step"] for v in val], [v["val_r"] for v in val], color="#6250d6", lw=1.2, ls="--", label="val recall")
            ax.set_ylim(0, 102)
            ax.set_title("F1 (%)")
            ax.legend(fontsize=7, loc="lower right", frameon=False)
        elif p == "val_loss":
            vv = [v for v in val if "val_loss" in v]
            ax.plot([v["step"] for v in vv], [v["val_loss"] for v in vv], color="#eb6834", lw=2, marker="o")
            ax.set_title("validation loss (per epoch)")
        elif p == "cos":
            ax.plot([d[0] for d in diag], [d[1] for d in diag], color="#1baf7a", lw=2, label="cosine to FP")
            if any(d[2] is not None for d in diag):
                ax.plot([d[0] for d in diag if d[2] is not None], [d[2] for d in diag if d[2] is not None], color="#eda100", lw=2, label="centred cosine")
            ax.set_title("encoder similarity to FP")
            ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle(name, fontweight="bold")
    fig.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    fig.savefig(Path(out) / f"{safe}.png", dpi=160)
    plt.close(fig)
    print(f"  {name}: {len(train)} train points, {len(val)} validation points, {len(diag)} cosine points -> {safe}.png")
    return train, diag, val


def plot_fp_csv(path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("  fp-csv: empty")
        return
    cols = list(rows[0].keys())
    print("  fp-csv columns:", cols)
    stepc = next((c for c in cols if "step" in c.lower()), None)
    f1c = next((c for c in cols if "f1" in c.lower()), None)
    if not (stepc and f1c):
        print("  fp-csv: could not find step / F1 columns; send me the column list above")
        return

    def num(x):
        try:
            return float(re.sub(r"[^0-9.\-]", "", str(x)))
        except ValueError:
            return None
    pts = sorted((num(r[stepc]), num(r[f1c])) for r in rows if num(r[stepc]) is not None and num(r[f1c]) is not None)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#eb6834", lw=2, marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel(f1c)
    ax.set_title("FP fine-tune: validation F1 at saved checkpoints")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececE8")
    fig.tight_layout()
    fig.savefig(Path(out) / "fp_leaderboard.png", dpi=160)
    plt.close(fig)
    print("  fp-csv -> fp_leaderboard.png")


def selftest():
    sample = (
        "Epoch [01/02] Step [0010/100] (Global 00010/00200) Loss: 5.0000 Acc: 10.00% F1: 20.00%\n"
        "[qat-diag] step ~  100 | cos(student,FP) 0.9425 | cos CENTERED 0.8465 | norm ratio 0.984\n"
        "Epoch [01/02] Step [0100/100] (Global 00100/00200) Loss: 1.0000 Acc: 90.00% F1: 95.00%\n"
        "==========\nEpoch 1/2 | Epoch-01 (Step 100/200)\nTrain Loss      : 1.0\nValidation Loss : 0.5000\n"
        "Validation Acc  : 97.18% (Val F1: 96.65% | P: 96.21% | R: 97.09% @ best threshold 0.70; predict errors: 0)\n"
    )
    t, d, v = parse(sample)
    assert t == [(10, 5.0, 10.0, 20.0), (100, 1.0, 90.0, 95.0)], t
    assert d == [(100, 0.9425, 0.8465)], d
    assert v == [{"epoch": 1, "step": 100, "val_loss": 0.5, "val_acc": 97.18, "val_f1": 96.65, "val_p": 96.21, "val_r": 97.09}], v
    with tempfile.TemporaryDirectory() as td:
        plot_log("selftest", sample, td)
        assert (Path(td) / "selftest.png").exists()
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=[], help='"name=path" (repeatable)')
    ap.add_argument("--fp-csv", help="checkpoint_leaderboard.csv of the FP run")
    ap.add_argument("--out", default="curves")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    allrows = []
    for spec in a.log:
        name, _, path = spec.partition("=")
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        train, diag, val = plot_log(name, text, a.out)
        allrows += [(name, "train", s, loss, f1, "", "") for s, loss, acc, f1 in train]
        allrows += [(name, "cos", s, "", "", c, cc if cc is not None else "") for s, c, cc in diag]
        allrows += [(name, "val", v["step"], v.get("val_loss", ""), v["val_f1"], v["val_p"], v["val_r"]) for v in val]
    if allrows:
        with open(Path(a.out) / "curves_data.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["log", "kind", "step", "loss", "f1_or_cos", "p_or_cos_centered", "r"])
            w.writerows(allrows)
    if a.fp_csv:
        plot_fp_csv(a.fp_csv, a.out)


if __name__ == "__main__":
    main()
