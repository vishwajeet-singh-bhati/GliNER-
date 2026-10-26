"""
How well does E8 quantization represent the REAL encoder weights, and would a random Hadamard rotation help?

For each of the 72 encoder Linear weights it reports
    kurt     excess kurtosis of the weights (0 = Gaussian; large = heavy tails / outliers)
    max/std  largest |w| in units of the row-std (outlier size)
    err      relative weight error  ||Q(W) - W|| / ||W||  with the repo's E8 quantizer
    err_rot  the same after a random-sign block-Hadamard rotation of the input dimension
             (orthogonal, so the error is directly comparable; this is the 'incoherence processing' of QuIP#)
and prints Gaussian reference values from the same quantizer (random matrices of the same shape).

How to read it
    err close to the Gaussian reference            -> the quantizer is doing about as well as E8 can; weights are not the issue
    err >> Gaussian reference, kurt large          -> heavy tails: a rotation (err_rot much lower) is worth building
    err_rot ~= err                                 -> a rotation would NOT help; look elsewhere

Usage (PowerShell, repo root):
    python diagnostics\\e8_weight_error.py --checkpoint "D:\\...\\checkpoint-6792"            # 1-bit codebook
    python diagnostics\\e8_weight_error.py --checkpoint "D:\\...\\checkpoint-6792" --bits 16  # 2-bit codebook (slower)
    python diagnostics\\e8_weight_error.py --selftest
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quantization"))
from e8_quant import E8QATLinear  # noqa: E402
from ptq_weight_only import classify  # noqa: E402


def hadamard(n: int) -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix (n a power of two); orthogonal."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def rotate_inputs(w: torch.Tensor, block: int = 256, seed: int = 0) -> torch.Tensor:
    """W R with R = blockdiag(H) * diag(random signs); needs in_features % block == 0."""
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (w.shape[1],), generator=g) * 2 - 1).to(w.dtype)
    h = hadamard(block).to(device=w.device, dtype=w.dtype)
    return ((w * signs.to(w.device)).reshape(w.shape[0], -1, block) @ h).reshape(w.shape)


@torch.no_grad()
def e8_rel_err(w: torch.Tensor, bits: int = 8) -> float:
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    lin.weight.data.copy_(w.float().cpu())
    m = E8QATLinear.from_linear(lin, scaling_mode="row_sigma_learnable", learnable=False, codebook_bits=bits)
    m = m.to(w.device).eval()
    q = m(torch.eye(w.shape[1], device=w.device)).t()
    return ((q - w).norm() / w.norm().clamp_min(1e-12)).item()


def stats(w: torch.Tensor):
    w = w.float()
    z = (w - w.mean(1, keepdim=True)) / w.std(1, keepdim=True).clamp_min(1e-12)
    return (z ** 4).mean().item() - 3.0, z.abs().max().item()


def selftest():
    torch.manual_seed(0)
    gauss = torch.randn(256, 256) * 0.04
    heavy = torch.distributions.StudentT(2.5).sample((256, 256)) * 0.03
    eg, eh = e8_rel_err(gauss), e8_rel_err(heavy)
    erg, erh = e8_rel_err(rotate_inputs(gauss)), e8_rel_err(rotate_inputs(heavy))
    print(f"gaussian: err {eg:.3f} -> rotated {erg:.3f} | heavy-tailed: err {eh:.3f} -> rotated {erh:.3f}")
    assert abs(eg - erg) < 0.03, "rotation should not change a Gaussian matrix's error much"
    assert erh < eh - 0.03, "rotation should reduce the error of a heavy-tailed matrix"
    r = rotate_inputs(heavy)
    assert abs(r.norm().item() - heavy.norm().item()) < 1e-3 * heavy.norm().item(), "rotation must be orthogonal"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="dir with pytorch_model.bin or model.safetensors (the FP checkpoint)")
    ap.add_argument("--bits", type=int, default=8, choices=[8, 16], help="8 = 1 bit/weight codebook, 16 = 2 bits/weight")
    ap.add_argument("--block", type=int, default=256, help="Hadamard block size (power of two dividing 768 and 3072)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.checkpoint:
        ap.error("--checkpoint is required")
    ck = Path(a.checkpoint)
    wf = next((ck / n for n in ("pytorch_model.bin", "model.safetensors") if (ck / n).exists()), None)
    if wf is None:
        raise FileNotFoundError(f"no pytorch_model.bin / model.safetensors in {ck}")
    if wf.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(wf), device="cpu")
    else:
        sd = torch.load(str(wf), map_location="cpu", weights_only=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    print(f"codebook: {'1' if a.bits == 8 else '2'} bit/weight | device {dev} | Hadamard block {a.block}")
    rows = []
    for k, t in sd.items():
        if classify(k, t) != "enc" or t.shape[1] % a.block or t.shape[1] < 64:
            continue
        w = t.float().to(dev)
        kurt, mx = stats(w)
        rows.append((k.split(".encoder.layer.")[-1], tuple(t.shape), kurt, mx, e8_rel_err(w, a.bits),
                     e8_rel_err(rotate_inputs(w, a.block), a.bits)))
    gref = {}
    for shape in {r[1] for r in rows}:
        g = torch.randn(*shape, device=dev) * 0.04
        gref[shape] = e8_rel_err(g, a.bits)

    print(f"\n{'layer':<46}{'shape':>14}{'kurt':>7}{'max/std':>9}{'err':>7}{'err_rot':>9}{'gauss ref':>10}")
    for name, shape, kurt, mx, e, er in rows:
        print(f"{name:<46}{str(shape):>14}{kurt:>7.1f}{mx:>9.1f}{e:>7.3f}{er:>9.3f}{gref[shape]:>10.3f}")
    n = len(rows)
    print(f"\n{n} layers | mean err {sum(r[4] for r in rows) / n:.3f} | mean err after rotation {sum(r[5] for r in rows) / n:.3f} "
          f"| mean Gaussian reference {sum(gref[r[1]] for r in rows) / n:.3f} | mean excess kurtosis {sum(r[2] for r in rows) / n:.1f}")


if __name__ == "__main__":
    main()
