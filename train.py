"""python train.py RANK -> out/sft_r{RANK} (LoRA adapter) + results/sft_r{RANK}_log.json"""
import json
import os
import sys

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from receipts import MODEL, load

r = int(sys.argv[1])
processor = AutoProcessor.from_pretrained(MODEL, fix_mistral_regex=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
# alpha = 2r keeps the update scale fixed, so only capacity differs between ranks; vision tower stays frozen
lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
args = SFTConfig(output_dir=f"out/sft_r{r}", num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                 learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05, max_length=None, bf16=True,
                 logging_steps=5, save_strategy="epoch", report_to="none", seed=0)
trainer = SFTTrainer(model=model, args=args, train_dataset=load("train"), processing_class=processor, peft_config=lora)
trainer.train()
trainer.save_model()
os.makedirs("results", exist_ok=True)
json.dump(trainer.state.log_history, open(f"results/sft_r{r}_log.json", "w"))
