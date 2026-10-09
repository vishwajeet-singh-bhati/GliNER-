import argparse
import json
import sys
import os
from pathlib import Path

# Fix HF Hub hanging on network requests (like the Mistral regex check)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from gliner import GLiNER

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quantization"))
from quantize_gliner import quantize_gliner_encoder

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--output-dir", default="models/antshield-gliner-v1")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-types", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--scaling-mode", default="block_max", choices=["block_max", "row_sigma", "row_sigma_learnable"])
    ap.add_argument("--no-learnable-scaling", dest="learnable_scaling", action="store_false")
    args = ap.parse_args()

    import torch
    use_bf16 = (not args.fp16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("Using bf16")
    precision_kwargs = {"bf16": True} if use_bf16 else {"fp16": args.fp16}

    train_data = load_jsonl(Path(args.data_dir) / "train_finetune.jsonl")
    val_data = load_jsonl(Path(args.data_dir) / "val_finetune.jsonl")

    model = GLiNER.from_pretrained(args.base_model, local_files_only=True)
    
    # Inject max_types directly into config to avoid the TrainingArguments TypeError
    if hasattr(model, "config"):
        model.config.max_types = args.max_types
        print(f"Set model.config.max_types = {args.max_types}")

    if args.quantize:
        print(f"Quantizing backbone encoder(s), scaling_mode={args.scaling_mode} ...")
        n = quantize_gliner_encoder(model, scaling_mode=args.scaling_mode, learnable=args.learnable_scaling)
        print(f"Total Linear layers quantized: {n}")

    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=val_data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        **precision_kwargs,
    )
    trainer.save_model()
    print(f"Saved fine-tuned model to {args.output_dir}")

if __name__ == "__main__":
    main()