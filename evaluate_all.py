import json
import time
from pprint import pprint
from gliner import GLiNER

# Load validation data
val_data = []
with open('data/val_finetune.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        val_data.append(json.loads(line))

# Only testing the fully stabilized final checkpoints
checkpoints = [12000, 12280]

for ckpt in checkpoints:
    path = f"D:/antshield-gliner-qat-balanced/checkpoint-{ckpt}"
    print(f"\n{'='*65}\nEvaluating Checkpoint: {ckpt}\n{'='*65}")
    
    try:
        print("[1/2] Loading 8-bit model weights...")
        model = GLiNER.from_pretrained(path, local_files_only=True)
        
        print(f"[2/2] Running evaluation on {len(val_data)} windows. Please wait ~1 minute...")
        
        # CAPTURE AND PRINT THE RESULTS!
        results = model.evaluate(val_data, flat_ner=True)
        
        print("\n🏆 RESULTS FOR CHECKPOINT", ckpt)
        # GLiNER usually returns a tuple (metrics_dict, f1_score) or just a dict
        if isinstance(results, tuple):
            pprint(results[0])
            print(f"\nOverall F1 Score: {results[1]}")
        else:
            pprint(results)
            
    except Exception as e:
        print(f"Failed to evaluate {ckpt}: {e}")
