"""python eval.py NAME [ADAPTER_DIR] -> results/NAME.json. Greedy decoding on the 100 CORD test receipts."""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from receipts import MODEL, load, score

name, adapter = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
B = int(os.environ.get("BATCH", 1))
test = load("test")
processor = AutoProcessor.from_pretrained(MODEL, fix_mistral_regex=True)
processor.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
if adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()

texts, t0 = [], time.time()
for i in range(0, len(test), B):
    rows = test[i : i + B]
    chats = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p[0]["content"]}]}] for p in rows["prompt"]]
    prompts = [processor.apply_chat_template(c, add_generation_prompt=True) for c in chats]
    batch = processor(text=prompts, images=[[im] for im in rows["image"]], return_tensors="pt", padding=True).to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=768, do_sample=False)
    texts += processor.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
secs = time.time() - t0

scores = np.array([score(t, g) for t, g in zip(texts, test["gt"])])  # columns: valid, f1
boot = [scores[np.random.default_rng(s).integers(0, len(scores), len(scores)), 1].mean() for s in range(1000)]
summary = {"name": name, "n": len(texts), "valid_json": scores[:, 0].mean(), "f1": scores[:, 1].mean(),
           "f1_ci95": [np.percentile(boot, 2.5), np.percentile(boot, 97.5)], "sec_per_receipt": secs / len(texts), "batch": B}
os.makedirs("results", exist_ok=True)
json.dump({"summary": summary, "outputs": texts, "scores": scores.tolist()}, open(f"results/{name}.json", "w"), indent=1, default=float)
print(json.dumps(summary, default=float))
