# Pipeline fixes and run order

## What was wrong -> what changed

| # | Cause of the bad numbers (P~0.2, R~0.8) | Fix | File |
|---|---|---|---|
| 1 | GLiNER prompts each training window only with labels present in it (+ random in-batch negatives); entity-free windows were prompted with a dummy "" label. The model never learned "label asked, nothing there". Reproduced against gliner 0.2.29. | Every window is prompted with ALL labels (`ner_labels`) | train/finetune_gliner.py (`LabelPromptDataset`) |
| 2 | Audit prompted 17 labels, one never trained | Eval uses `train_labels.json` written by training | eval/evaluate_gliner_audit.py |
| 3 | Audit chunker counted GLiNER max_len (words) as subwords -> split ~half the windows | Word-based chunking (a 300-word window = 1 pass) | eval/evaluate_gliner_audit.py |
| 4 | Trainer "eval" used test file, positives only | Real val split (whole apps), natural mix of positive/negative windows; refuses to fall back to test | data/build_dataset.py, train/finetune_gliner.py |
| 5 | Train prior 51% positive vs test 14.5% | neg-ratio 3.0 + test/val exact-duplicate removal: 35% vs 20% | data/build_dataset.py |
| 6 | Masked advertiser ids were not v4 UUIDs (real unlabeled UUIDs are 84.5% v4) -> a one-character shortcut | v4 UUIDs; Luhn IMEI/ICCID; MCC/MNC IMSI; lat/lon-aware location | data/pii_shapes.py, recon_synthesizer.py |
| 7 | 7.5% of spans glued to neighbours / mid-token | Boundary-safe masking + strict span mapping: 0 misaligned of 16,320 | prepare_finetune_data.py, gliner_tokenizer.py |
| 8 | Threshold never tuned; fixed 0.5 | Chosen on val, applied once to test; sweep table reported | eval/evaluate_gliner_audit.py |
| 9 | Heads trained at encoder LR 5e-6; E8 scale params effectively frozen | `--others-lr 5e-5`, `--qat-scale-lr 1e-3` (verified: theta moves 1e-3/step) | train/finetune_gliner.py |
| 10 | QAT from the pretrained weights with ~1 bit/weight and no FP baseline | Train FP first, then `--init-from` it for QAT | train/finetune_gliner.py |
| 11 | 3,026 identical NUL-blob test windows + many duplicates; template twins across splits | exact-dup removal in val/test, canonical-twin guard train<->val<->test | data/build_dataset.py |
| 12 | Record ids contained absolute paths -> different data on another machine | relative ids, sorted file order | antshield_loader.py |

## Run order (PowerShell, from the repo root)

    python data/test_data_pipeline.py
    python eval/test_evaluate_gliner_audit.py
    python data/build_dataset.py --dataset-root <path>\antshield_public_dataset --output-dir data --force
    python data/validate_dataset.py --data-dir data

    # 1) full-precision fine-tune FIRST (this is your reference)
    python train/finetune_gliner.py --data-dir data --output-dir models\fp-v2 --max-steps 3000 --batch-size 1 --grad-accum 8 --check-prompts
    python train/finetune_gliner.py --data-dir data --output-dir models\fp-v2 --max-steps 3000 --batch-size 1 --grad-accum 8

    # 2) evaluate: threshold picked on val, applied once to test
    python eval/evaluate_gliner_audit.py --models models\fp-v2\best_checkpoint --data-path data\test_finetune.jsonl --val-path data\val_finetune.jsonl --output-dir eval_results\fp-v2

    # 3) only then QAT, starting from the FP model
    python train/finetune_gliner.py --data-dir data --output-dir models\qat-v2 --quantize --scaling-mode row_sigma_learnable --init-from models\fp-v2\best_checkpoint --max-steps 3000 --learning-rate 5e-6 --others-lr 2e-5
    python eval/evaluate_gliner_audit.py --quantized --scaling-mode row_sigma_learnable --models models\qat-v2\best_checkpoint --data-path data\test_finetune.jsonl --val-path data\val_finetune.jsonl --output-dir eval_results\qat-v2

If you hit CUDA OOM, add `--max-window-tokens 256` to training (old crop behaviour).

## Not verified here (no GPU, no HF weights in the sandbox)
No new F1 numbers exist yet. Verified: data build + validator, alignment, unit tests, the collator prompt
change against gliner 0.2.29, the E8 optimizer group.
Not run: end-to-end training, batched `model.inference` in the evaluator (falls back automatically), QAT eval.
Known limits: val is app-disjoint from train but test shares ~72% of apps with train, so val is a bit harder
than test; 7 classes (email, gender, imsi, last name, password, phone, iccid) have <=36 real test spans and are
trained on synthetic values, so their per-class F1 is noisy.
