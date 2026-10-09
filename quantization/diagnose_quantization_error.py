"""
No-training diagnostic: measures how much damage E8 quantization actually
does to GLiNER REAL pretrained weights, before any fine-tuning touches
them. Takes seconds, not hours -- run this before another training attempt.

Why this matters: the quantization code (e8_quant.py) is proven correct in
isolation (it already works for BERT, achieving ~0.35 F1-point gap vs FP32).
But its constants (BLOCK_SIZE=32, the E8 codebook, ROW_SIGMA_CB=5.0) were
calibrated against BERT-base weight distribution. Nobody has checked
whether they fit mDeBERTa-v3. If they do not, the SAME correct code could
be producing much larger reconstruction error on this checkpoint -- and
since block_max/row_sigma both recompute scale fresh from the CURRENT weight
on every forward pass, bad bin-assignment stability could compound with
gradient descent in a way that looks exactly like non-convergent training
loss, without there being a bug anywhere.

What this reports, per scaling mode, across all 72 backbone Linear layers,
on the untouched pretrained weights:
  - relative Frobenius reconstruction error per layer (||W - W_quant|| / ||W||)
  - fraction of subvectors landing on the all-zero codeword (the exact
    failure pattern already found and fixed once before for
    row_sigma_learnable, per docs/HISTORY.md -- worth checking whether a
    similar collapse is happening here too, for a different reason)

Rule of thumb: healthy 8-bit-ish quantization usually sits in the few-percent
relative error range. Numbers in the 30-80%+ range, or a zero-codeword
fraction anywhere near 0.5+, indicate the scaling constants do not fit this
checkpoint weight distribution and need retuning before more training time
is spent.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_quant import build_e8_codebook_8bit, e8_quantize_ste, BLOCK_SIZE, SUBVEC_DIM
from quantize_gliner import get_backbone_encoders


def zero_codeword_fraction(W: torch.Tensor, codebook: torch.Tensor, norms: torch.Tensor,
                            scale: torch.Tensor) -> float:
    """What fraction of 8-D subvectors quantize to the all-zero codeword."""
    import torch.nn.functional as F
    rows, cols = W.shape
    pad = (-cols) % BLOCK_SIZE
    W_padded = F.pad(W, (0, pad)) if pad else W
    blocks = W_padded.reshape(-1, BLOCK_SIZE)
    norm_blocks = blocks / scale
    subvecs = norm_blocks.reshape(-1, SUBVEC_DIM)
    dots = torch.einsum("nd,cd->nc", subvecs, codebook)
    dists = norms.view(1, -1) - 2 * dots
    best_idx = dists.argmin(dim=-1)
    zero_idx = int(norms.argmin().item())
    return (best_idx == zero_idx).float().mean().item()


def diagnose_layer(W: torch.Tensor, codebook: torch.Tensor, norms: torch.Tensor, scaling_mode: str):
    W_quant = e8_quantize_ste(W, codebook, norms, scaling_mode)
    rel_error = (W - W_quant).norm().item() / (W.norm().item() + 1e-12)

    # recompute the same scale the quantizer used, to check zero-codeword fraction
    if scaling_mode == "block_max":
        rows, cols = W.shape
        pad = (-cols) % BLOCK_SIZE
        import torch.nn.functional as F
        W_padded = F.pad(W, (0, pad)) if pad else W
        blocks = W_padded.reshape(-1, BLOCK_SIZE)
        max_val = blocks.abs().amax(dim=1, keepdim=True)
        scale = torch.where(max_val > 1e-8, max_val / 2.0, torch.full_like(max_val, 1e-8))
    else:  # row_sigma
        from e8_quant import ROW_SIGMA_CB, get_row_sigma_ymax
        rows, cols = W.shape
        pad = (-cols) % BLOCK_SIZE
        import torch.nn.functional as F
        W_padded = F.pad(W, (0, pad)) if pad else W
        row_sigma = W_padded.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
        bpr = W_padded.shape[1] // BLOCK_SIZE
        row_scale = (ROW_SIGMA_CB * row_sigma) / get_row_sigma_ymax()
        scale = row_scale.repeat_interleave(bpr, dim=0)

    zero_frac = zero_codeword_fraction(W, codebook, norms, scale)
    return rel_error, zero_frac


def main():
    from gliner import GLiNER

    base_model = sys.argv[1] if len(sys.argv) > 1 else "urchade/gliner_multi_pii-v1"
    print(f"Loading {base_model} (unquantized) ...")
    model = GLiNER.from_pretrained(base_model)

    codebook = build_e8_codebook_8bit()
    norms = (codebook ** 2).sum(dim=1)

    import torch.nn as nn
    for scaling_mode in ["block_max", "row_sigma"]:
        print(f"\n=== scaling_mode={scaling_mode} ===")
        errors, zero_fracs = [], []
        for name, encoder in get_backbone_encoders(model):
            for layer_name, module in encoder.named_modules():
                if isinstance(module, nn.Linear) and module.in_features >= 64:
                    with torch.no_grad():
                        rel_error, zero_frac = diagnose_layer(
                            module.weight.float(), codebook, norms, scaling_mode)
                    errors.append(rel_error)
                    zero_fracs.append(zero_frac)

        errors_t = torch.tensor(errors)
        zero_t = torch.tensor(zero_fracs)
        print(f"  layers checked: {len(errors)}")
        print(f"  relative reconstruction error: min={errors_t.min():.3f} "
              f"median={errors_t.median():.3f} max={errors_t.max():.3f}")
        print(f"  zero-codeword fraction:        min={zero_t.min():.3f} "
              f"median={zero_t.median():.3f} max={zero_t.max():.3f}")


if __name__ == "__main__":
    main()