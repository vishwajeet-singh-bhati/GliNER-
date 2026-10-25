"""
QAT sensitivity probe: how much does E8 quantization change the encoder's output?

Why: the QAT runs from the fine-tuned FP model start at loss ~7 and never learn (accuracy 0.00%).
That is either (a) 1-bit-per-weight is simply too destructive when applied to every layer at once,
or (b) there is a bug in the quantized layer / integration.  This probe separates the two WITHOUT any
training: it quantizes different parts of the pretrained encoder and measures how far the encoder
output moves from the FP output.

Reading the table (cos = cosine similarity of the final hidden state vs the FP encoder, 1.0 = identical):
  * "none"                    must be 1.000 (sanity check of the probe itself).
  * a SINGLE block quantized  should stay fairly high (roughly >0.8).  If even one block gives ~0 or NaN
                              -> that points to a bug (scale, dtype, integration), not "too many layers".
  * "all" much lower than any single block, falling smoothly layer by layer
                              -> error compounding through depth: intrinsic to quantizing everything at 1 bit.
  * "all (bf16 autocast)"     differing a lot from "all" (or non-finite) -> bf16 numerical problem.

Usage (PowerShell, venv active, repo root):
  python diagnostics\\qat_sensitivity_probe.py --init-from D:\\gliner-antshield\\models\\fp-v2_6800steps_2026-10-24\\checkpoint-6792
  python diagnostics\\qat_sensitivity_probe.py --selftest        # tiny random model, no downloads
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quantization"))
from e8_quant import E8QATLinear  # noqa: E402
from ptq_weight_only import fake_quant  # noqa: E402


# ---------------------------------------------------------------- helpers (pure, unit-testable)
def quantize_selected(hf_model: nn.Module, pred, scaling_mode: str, min_features: int = 64, codebook_bits: int = 8) -> int:
    """Replace the nn.Linear layers of hf_model.encoder.layer[i] for which pred(i, qualified_name) is True."""
    n = 0
    for i, layer in enumerate(hf_model.encoder.layer):
        for qual, mod in list(layer.named_modules()):
            if not isinstance(mod, nn.Linear) or mod.in_features < min_features:
                continue
            if not pred(i, qual):
                continue
            parent_name, _, child = qual.rpartition(".")
            parent = layer.get_submodule(parent_name) if parent_name else layer
            new = E8QATLinear.from_linear(mod, scaling_mode=scaling_mode, learnable=False, codebook_bits=codebook_bits)
            new = new.to(device=mod.weight.device, dtype=mod.weight.dtype)   # from_linear builds on the CPU
            setattr(parent, child, new)
            n += 1
    return n


@torch.no_grad()
def fake_quant_encoder(hf_model: nn.Module, bits: int, group_size: int = 64, min_features: int = 64) -> int:
    """Scalar (int-b, per-group) round-to-nearest of all encoder Linear weights, in place. For same-bit-budget comparison."""
    n = 0
    for layer in hf_model.encoder.layer:
        for mod in layer.modules():
            if isinstance(mod, nn.Linear) and mod.in_features >= min_features:
                mod.weight.data = fake_quant(mod.weight.data, bits, group_size)
                n += 1
    return n


@torch.no_grad()
def run_hidden(hf_model, ids, mask, bf16=False):
    ctx = torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=bf16)
    with ctx:
        out = hf_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    return [h.float() for h in out.hidden_states]


def compare(h_fp, h_q, mask):
    """Per hidden-state layer: (cosine, relative error, norm ratio, all finite) over non-pad tokens."""
    m = mask.bool()
    rows = []
    for a, b in zip(h_fp, h_q):
        a, b = a[m], b[m]
        finite = bool(torch.isfinite(b).all().item())
        b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        rel = ((b - a).norm() / a.norm().clamp_min(1e-12)).item()
        ratio = (b.norm() / a.norm().clamp_min(1e-12)).item()
        rows.append((cos, rel, ratio, finite))
    return rows


def make_configs(n_layers: int):
    half = n_layers // 2
    mid = n_layers // 2
    return [
        ("none (sanity)", lambda i, n: False),
        (f"1 block only (block {mid})", lambda i, n: i == mid),
        ("1 block only (block 0)", lambda i, n: i == 0),
        (f"1 block only (block {n_layers - 1})", lambda i, n: i == n_layers - 1),
        (f"first {half} blocks", lambda i, n: i < half),
        (f"last {n_layers - half} blocks", lambda i, n: i >= half),
        ("attention matrices only", lambda i, n: n.startswith("attention")),
        ("feed-forward only", lambda i, n: not n.startswith("attention")),
        ("all", lambda i, n: True),
    ]


def make_hybrid_configs(n_layers: int):
    """Quantize everything EXCEPT the most sensitive parts (kept in full precision)."""
    out = []
    for k in (1, 2, 3, 4):
        out.append((f"all except last {k} block(s)", lambda i, n, k=k: i < n_layers - k))
    out.append(("all except FFN of last 4 blocks",
                lambda i, n: not (i >= n_layers - 4 and not n.startswith("attention"))))
    out.append(("attention in all blocks + FFN in first 6",
                lambda i, n: n.startswith("attention") or i < 6))
    out.append(("first 4 blocks only", lambda i, n: i < 4))
    return out


def run_probe(hf_model, ids, mask, scaling_mode, log=print, extended=False, codebook_bits=8, scalar_bits=()):
    hf_model.eval()
    h_fp = run_hidden(hf_model, ids, mask)
    results = []
    if scalar_bits:
        for b in scalar_bits:
            q = copy.deepcopy(hf_model)
            n = fake_quant_encoder(q, b, 64)
            q.eval()
            per_layer = compare(h_fp, run_hidden(q, ids, mask), mask)
            results.append((f"scalar int{b} (groups of 64), all", n, per_layer))
            cos, rel, ratio, fin = per_layer[-1]
            log(f"{f'scalar int{b} g64, all 72 (reference)':<34}{n:>4} linears | cos {cos:7.3f} | rel.err {rel:7.3f} | norm ratio {ratio:6.2f} | finite {fin}")
            del q
    configs = make_configs(len(hf_model.encoder.layer)) + [("all (bf16 autocast)", lambda i, n: True)]
    if extended:
        configs += make_hybrid_configs(len(hf_model.encoder.layer))
    for name, pred in configs:
        q = copy.deepcopy(hf_model)
        n = quantize_selected(q, pred, scaling_mode, codebook_bits=codebook_bits)
        q.eval()
        h_q = run_hidden(q, ids, mask, bf16=name.endswith("(bf16 autocast)"))
        per_layer = compare(h_fp, h_q, mask)
        results.append((name, n, per_layer))
        cos, rel, ratio, fin = per_layer[-1]
        log(f"{name:<34}{n:>4} linears | cos {cos:7.3f} | rel.err {rel:7.3f} | norm ratio {ratio:6.2f} | finite {fin}")
        del q
        if ids.is_cuda:
            torch.cuda.empty_cache()
    return results


# ------------------------------------------------------------------------------- entry points
def selftest():
    from transformers import DebertaV2Config, DebertaV2Model
    torch.manual_seed(0)
    cfg = DebertaV2Config(vocab_size=200, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                          intermediate_size=128, relative_attention=True, position_buckets=16,
                          max_relative_positions=-1, pos_att_type=["p2c", "c2p"], max_position_embeddings=128,
                          position_biased_input=False, share_att_key=True, norm_rel_ebd="layer_norm")
    m = DebertaV2Model(cfg)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0, 0.04)
    ids = torch.randint(5, 200, (4, 24))
    mask = torch.ones_like(ids)
    mask[:, 20:] = 0
    res = run_probe(m, ids, mask, "row_sigma_learnable", extended=True, scalar_bits=(8, 4, 2))
    by = {r[0]: r for r in res}
    assert by["none (sanity)"][2][-1][0] > 0.9999, "sanity row must be identical to FP"
    assert by["all"][2][-1][3], "outputs must be finite"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--init-from", default=None, help="checkpoint dir containing pytorch_model.bin")
    ap.add_argument("--data-file", default=str(ROOT / "data" / "val_finetune.jsonl"))
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--scaling-mode", default="row_sigma_learnable")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--codebook-bits", type=int, default=8, choices=[8, 16], help="8 = current 1 bit/weight codebook, 16 = 2 bits/weight")
    ap.add_argument("--scalar-bits", type=int, nargs="*", default=[], help="also print plain scalar int-b quantization rows (e.g. 8 4 3 2) as a same-bit-budget reference")
    ap.add_argument("--extended", action="store_true", help="also test hybrid configs that keep sensitive parts in FP")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    from gliner import GLiNER
    model = GLiNER.from_pretrained(args.base_model, local_files_only=True)
    if args.init_from:
        sd = torch.load(str(Path(args.init_from) / "pytorch_model.bin"), map_location="cpu")
        target = model.model
        keys = set(target.state_dict().keys())
        direct = len(keys & set(sd))
        stripped_sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        stripped = len(keys & set(stripped_sd))
        use = sd if direct >= stripped else stripped_sd
        target.load_state_dict(use, strict=False)
        print(f"init-from: matched {max(direct, stripped)}/{len(keys)} tensors")
    hf = model.model.token_rep_layer.bert_layer.model
    tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)

    texts = []
    with open(args.data_file, encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            if w.get("ner"):
                texts.append(" ".join(w["tokenized_text"]))
            if len(texts) >= args.n_windows:
                break
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    hf = hf.float().to(dev)
    ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    print(f"probe on {len(texts)} windows, {int(mask.sum())} tokens, device={dev}, layers={len(hf.encoder.layer)}")
    print(f"codebook_bits={args.codebook_bits} ({'2' if args.codebook_bits == 16 else '1'} bit/weight)")
    results = run_probe(hf, ids, mask, args.scaling_mode, extended=args.extended, codebook_bits=args.codebook_bits, scalar_bits=args.scalar_bits)
    allrow = next(r for r in results if r[0] == "all")
    print("\nper-layer cosine for 'all' (embedding output, then blocks 1..N):")
    print("  " + " ".join(f"{c:.2f}" for c, _, _, _ in allrow[2]))


if __name__ == "__main__":
    main()
