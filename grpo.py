"""python grpo.py STEPS -> out/grpo_STEPS (LoRA adapter on top of merged out/sft_r8) + results/grpo_STEPS_log.json. Resumes from checkpoints."""
import glob
import json
import os
import sys

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from receipts import MODEL, load, reward

steps, out = int(sys.argv[1]), f"out/grpo_{sys.argv[1]}"
processor = AutoProcessor.from_pretrained(MODEL, fix_mistral_regex=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "out/sft_r8").merge_and_unload()  # start from the SFT model; the new LoRA sits on top
lora = LoraConfig(r=8, lora_alpha=16, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
# one receipt x 8 samples per step: the processor can't stack images of different sizes, so a batch must be one receipt
# no KL term (beta=0, as in Magistral), so no reference model is held
args = GRPOConfig(output_dir=out, max_steps=steps, per_device_train_batch_size=8, gradient_accumulation_steps=1, num_generations=8,
                  max_completion_length=768, beta=0.0, learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=0.05,
                  bf16=True, logging_steps=1, save_steps=25, save_total_limit=2, report_to="none", seed=0)
trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args, train_dataset=load("train"), processing_class=processor, peft_config=lora)
trainer.train(resume_from_checkpoint=bool(glob.glob(f"{out}/checkpoint-*")))
trainer.save_model()
os.makedirs("results", exist_ok=True)
json.dump(trainer.state.log_history, open(f"results/grpo_{steps}_log.json", "w"))
