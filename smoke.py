"""python smoke.py data|tokens|sft|grpo"""
import sys

from receipts import MODEL, load, reward

stage = sys.argv[1]
train = load("train", 50)

if stage == "data":
    ex = train[0]
    print(ex["prompt"], ex["completion"][0]["content"][:200], ex["image"].size, sep="\n")
    sys.exit()

import torch  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

processor = AutoProcessor.from_pretrained(MODEL, fix_mistral_regex=True)
if stage == "tokens":
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": train[0]["prompt"][0]["content"]}]}]
    text = processor.apply_chat_template(msg, add_generation_prompt=True)
    print("prompt tokens incl. image:", processor(text=text, images=train[0]["image"], return_tensors="pt")["input_ids"].shape[1])
    sys.exit()

model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
lora = LoraConfig(r=16, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")

if stage == "sft":
    args = SFTConfig(output_dir="out/smoke_sft", max_steps=20, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                     learning_rate=2e-4, max_length=None, bf16=True, logging_steps=1, save_strategy="no", report_to="none")
    SFTTrainer(model=model, args=args, train_dataset=train, processing_class=processor, peft_config=lora).train()
elif stage == "grpo":
    args = GRPOConfig(output_dir="out/smoke_grpo", max_steps=5, per_device_train_batch_size=4, num_generations=4,
                      max_completion_length=512, beta=0.0, learning_rate=1e-5, bf16=True, logging_steps=1,
                      save_strategy="no", report_to="none")
    GRPOTrainer(model=model, reward_funcs=reward, args=args, train_dataset=train, processing_class=processor,
                peft_config=lora).train()
