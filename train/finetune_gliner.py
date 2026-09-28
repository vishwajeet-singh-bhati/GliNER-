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
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    train_data = load_jsonl(Path(args.data_dir) / "train_gliner.jsonl")
    val_data = load_jsonl(Path(args.data_dir) / "val_gliner.jsonl")

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
    )

    trainer.save_model()
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
