"""
Weight-only post-training quantization (no training) for a GLiNER checkpoint, to get an edge-size/quality
baseline that does NOT depend on QAT.

It "fake-quantizes" weights: each weight matrix is rounded to a b-bit symmetric grid (per output row, or per
group of `--group-size` input columns) and written back as ordinary float values.  The resulting checkpoint
therefore loads in the existing evaluator unchanged, and its accuracy is exactly what an int-b deployment of
those weights would give (weights only; activations stay float).  The report also prints the on-disk size an
edge deployment would need.

Three tensor groups, each with its own bit-width (0 = leave in float):
  --enc-bits    the 72 transformer Linear weights  (what E8 QAT targets, 85M params)
  --emb-bits    embedding tables                   (192M params: the biggest part of the model)
  --other-bits  every other 2-D weight (heads, LSTM, projections)

Size accounting: quantized tensors = numel*bits/8 + one fp16 scale per row/group; every tensor left in float
(biases, norms, groups set to 0) is counted as fp16.  FP32 total is printed for reference.

Examples (PowerShell):
  python quantization\\ptq_weight_only.py --checkpoint D:\\...\\checkpoint-6792 --out-dir D:\\...\\ptq-int8 --enc-bits 8 --emb-bits 8 --other-bits 8
  python quantization\\ptq_weight_only.py --selftest
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch


def classify(key: str, t: torch.Tensor) -> str:
    """'enc' | 'emb' | 'other' for 2-D weights, 'skip' for everything else."""
    if t.dim() != 2 or not key.endswith(".weight") or not t.is_floating_point():
        return "skip"
    if "embeddings" in key:
        return "emb"
    if ".encoder.layer." in key:
        return "enc"
    return "other"


def fake_quant(w: torch.Tensor, bits: int, group_size: int = 0) -> torch.Tensor:
    """Symmetric round-to-nearest, per row (group_size=0) or per group of columns."""
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    g = cols if group_size <= 0 or group_size >= cols else group_size
    pad = (-cols) % g
    wp = torch.nn.functional.pad(w.float(), (0, pad)) if pad else w.float()
    wg = wp.reshape(rows, -1, g)
    scale = (wg.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.round(wg / scale).clamp(-qmax, qmax)
    out = (q * scale).reshape(rows, -1)
    return (out[:, :cols] if pad else out).to(w.dtype)


def n_scales(shape, group_size: int) -> int:
    rows, cols = shape
    g = cols if group_size <= 0 or group_size >= cols else group_size
    return rows * -(-cols // g)


def quantize_state_dict(sd, bits, group_size=0):
    """bits: {'enc':int,'emb':int,'other':int}.  Returns (new_sd, report dict)."""
    new_sd, rep = {}, {k: {"params": 0, "bytes": 0.0, "rel_err_sum": 0.0, "tensors": 0} for k in ("enc", "emb", "other", "float")}
    fp32_bytes = 0.0
    for k, t in sd.items():
        fp32_bytes += t.numel() * 4
        grp = classify(k, t)
        b = bits.get(grp, 0) if grp != "skip" else 0
        if grp != "skip" and b > 0:
            new_t = fake_quant(t, b, group_size)
            err = ((new_t.float() - t.float()).norm() / t.float().norm().clamp_min(1e-12)).item()
            r = rep[grp]
            r["params"] += t.numel()
            r["bytes"] += t.numel() * b / 8 + n_scales(t.shape, group_size) * 2
            r["rel_err_sum"] += err
            r["tensors"] += 1
            new_sd[k] = new_t
        else:
            r = rep["float"]
            r["params"] += t.numel()
            r["bytes"] += t.numel() * 2          # kept in fp16 on the edge
            new_sd[k] = t
    rep["fp32_total_bytes"] = fp32_bytes
    return new_sd, rep


def print_report(rep, bits, group_size):
    mb = lambda x: x / 1e6
    print(f"\nbits: {bits} | group-size: {group_size or 'per-row'}")
    print(f"{'group':<8}{'params(M)':>10}{'size(MB)':>10}{'mean rel.err':>14}")
    total = 0.0
    for g in ("enc", "emb", "other", "float"):
        r = rep[g]
        err = f"{r['rel_err_sum'] / r['tensors']:.4f}" if r["tensors"] else "-"
        print(f"{g:<8}{r['params'] / 1e6:>10.1f}{mb(r['bytes']):>10.1f}{err:>14}")
        total += r["bytes"]
    print(f"TOTAL edge size: {mb(total):.0f} MB   (FP32 original: {mb(rep['fp32_total_bytes']):.0f} MB, "
          f"{rep['fp32_total_bytes'] / total:.1f}x smaller)")
    return total


def selftest():
    torch.manual_seed(0)
    sd = {
        "m.token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight": torch.randn(500, 64) * 0.05,
        "m.token_rep_layer.bert_layer.model.encoder.layer.0.attention.self.query_proj.weight": torch.randn(64, 64) * 0.04,
        "m.token_rep_layer.bert_layer.model.encoder.layer.0.attention.self.query_proj.bias": torch.zeros(64),
        "m.token_rep_layer.bert_layer.model.encoder.layer.0.output.LayerNorm.weight": torch.ones(64),
        "m.span_rep_layer.project.weight": torch.randn(64, 64) * 0.04,
    }
    new, rep = quantize_state_dict(sd, {"enc": 8, "emb": 8, "other": 0})
    assert rep["enc"]["tensors"] == 1 and rep["emb"]["tensors"] == 1 and rep["other"]["tensors"] == 0
    assert rep["enc"]["rel_err_sum"] < 0.02, "int8 should be near-lossless"
    new4, rep4 = quantize_state_dict(sd, {"enc": 4, "emb": 4, "other": 4}, group_size=32)
    assert rep4["enc"]["rel_err_sum"] > rep["enc"]["rel_err_sum"], "int4 must be lossier than int8"
    assert torch.equal(new["m.span_rep_layer.project.weight"], sd["m.span_rep_layer.project.weight"])  # other=0 untouched
    q = new["m.token_rep_layer.bert_layer.model.encoder.layer.0.attention.self.query_proj.weight"]
    assert q.shape == (64, 64) and len(torch.unique(q[0])) <= 255
    print_report(rep, {"enc": 8, "emb": 8, "other": 0}, 0)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="dir containing pytorch_model.bin (or model.safetensors)")
    ap.add_argument("--out-dir", help="output dir; put it INSIDE the training output dir so the evaluator finds gliner_config.json / train_labels.json")
    ap.add_argument("--enc-bits", type=int, default=8)
    ap.add_argument("--emb-bits", type=int, default=8)
    ap.add_argument("--other-bits", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=0, help="0 = per output row; e.g. 64 or 128 for group-wise")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.checkpoint or not args.out_dir:
        ap.error("--checkpoint and --out-dir are required")
    ck = Path(args.checkpoint)
    wf = next((ck / n for n in ("pytorch_model.bin", "model.safetensors") if (ck / n).exists()), None)
    if wf is None:
        raise FileNotFoundError(f"no pytorch_model.bin / model.safetensors in {ck}")
    if wf.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(wf), device="cpu")
    else:
        sd = torch.load(str(wf), map_location="cpu", weights_only=True)
    bits = {"enc": args.enc_bits, "emb": args.emb_bits, "other": args.other_bits}
    new_sd, rep = quantize_state_dict(sd, bits, args.group_size)
    total = print_report(rep, bits, args.group_size)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, str(out / "pytorch_model.bin"))
    # Copy every small non-weight file (config, tokenizer files with GLiNER's added special tokens, labels):
    # without them GLiNER rebuilds a plain tokenizer and refuses to load.
    skip = {"pytorch_model.bin", "model.safetensors", "optimizer.pt", "scheduler.pt", "training_args.bin",
            "scaler.pt", "ptq_report.json"}
    copied = []
    for f in ck.iterdir():
        if f.is_file() and f.name not in skip and not f.name.startswith("rng_state") and f.stat().st_size < 64e6:
            shutil.copy(f, out / f.name)
            copied.append(f.name)
    for extra in ("train_labels.json", "gliner_config.json"):      # sometimes only in the parent run dir
        if extra not in copied and (ck.parent / extra).exists():
            shutil.copy(ck.parent / extra, out / extra)
            copied.append(extra)
    print("copied alongside the weights:", ", ".join(sorted(copied)) or "(nothing)")
    (out / "ptq_report.json").write_text(json.dumps({
        "bits": bits, "group_size": args.group_size, "edge_size_mb": round(total / 1e6, 1),
        "fp32_size_mb": round(rep["fp32_total_bytes"] / 1e6, 1),
        "groups": {g: {"params_m": round(rep[g]["params"] / 1e6, 2), "size_mb": round(rep[g]["bytes"] / 1e6, 2)}
                   for g in ("enc", "emb", "other", "float")}}, indent=2), encoding="utf-8")
    print(f"\nsaved fake-quantized checkpoint -> {out}")


if __name__ == "__main__":
    main()
