# Mobile inference plan (Phase 2-3, not started)

Reuse from the BERT quantization project:
- ARM NEON kernels for the transformer encoder forward pass (FP32 / Q4_0 / E8)
- Wireless ADB benchmarking harness on the physical device
- DVFS warmup handling, streaming loader

New work needed for GLiNER specifically:
1. **Label encoding.** GLiNER encodes label prompts through the same backbone.
   At inference time on-device you'll want to precompute and cache label
   embeddings once (labels are fixed at ~19 AntShield types), not recompute
   them per packet -- this is a real latency win over naive port.
2. **Span representation.** Need start/end token pooling over encoder outputs
   to build span vectors -- a small addition to the kernel pipeline, not a
   rewrite.
3. **Span scoring head.** Bilinear/dot-product between span vectors and cached
   label embeddings, then thresholding. This is new C++ code, not present in
   the binary classifier kernel at all.
4. **Quantization target.** Decide backbone size before exporting -- check
   which backbone the fine-tuned checkpoint from Phase 1 ends up using
   (`gliner_multi_pii-v1` fine-tunes `gliner_multi-v2.1`, backbone size TBD
   pending your Phase 1 run) so the E8/Q4_0 kernel dimensions match.

Blocked on Phase 1 (fine-tuning) completing first, since we quantize whatever
backbone that phase settles on.
