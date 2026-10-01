"""
Phase 1: fine-tune GLiNER on AntShield so it gets used to this data's shape
(form-encoded/JSON packet bodies, not natural-language sentences) rather than
relying on zero-shot, which was trained on natural sentences and has no
reason to expect this input distribution.

Needs: pip install "gliner[training]"

Run data/prepare_finetune_data.py first to produce train_gliner.jsonl and
val_gliner.jsonl in --data-dir.
"""
import argparse
import json
from pathlib import Path

from gliner import GLiNER


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="dir containing train_gliner.jsonl / val_gliner.jsonl")
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--output-dir", default="models/antshield-gliner-v1")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument(
        "--batch-size", type=int, default=2,
        help="per-device batch size. Kept low by default (2) because this dataset's "
             "sequences run up to the model's 384-token limit, and max_width (the "
             "model's span-enumeration width) can't be reduced for a pretrained "
             "checkpoint without breaking weight compatibility -- so batch size is "
             "the main memory lever available here.",
    )
    ap.add_argument(
        "--grad-accum", type=int, default=4,
        help="gradient accumulation steps. Effective batch size = batch-size * grad-accum. "
             "Default 2*4=8 matches a typical GLiNER fine-tune effective batch size "
             "without the per-step memory spike of a literal batch of 8.",
    )
    ap.add_argument(
        "--fp16", action="store_true", default=True,
        help="mixed precision (default on) -- roughly halves activation memory on most GPUs.",
    )
    ap.add_argument(
        "--no-fp16", dest="fp16", action="store_false",
        help="disable mixed precision (e.g. if your GPU doesn't support it well).",
    )
    args = ap.parse_args()

    train_data = load_jsonl(Path(args.data_dir) / "train_finetune.jsonl")
    val_data = load_jsonl(Path(args.data_dir) / "val_finetune.jsonl")

    print(f"Train: {len(train_data)} windows, Val: {len(val_data)} windows")
    print(f"Loading base model {args.base_model} ...")
    model = GLiNER.from_pretrained(args.base_model)

    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=val_data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=args.fp16,
    )

    trainer.save_model()
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
