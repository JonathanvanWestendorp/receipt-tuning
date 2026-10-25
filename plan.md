# Ministral 3B on receipts: prompt, fine-tune, measure

Story in one line: a small open model, fine-tuned for a narrow job, against the same model prompted and a large API model, on accuracy, cost and speed.

## Setup
- Model: `mistralai/Ministral-3-3B-Instruct-2512-BF16` (Apache 2.0, 3.4B LM + 0.4B vision encoder). Starting from Instruct makes before/after a fair
  comparison: the "before" row is a model that can follow the prompt. Base-2512 is an optional ablation.
- Data: CORD-v2 (CC BY 4.0), 800 train / 100 validation / 100 test. Receipt image -> JSON (menu, sub_total, total). Images capped at 1024 px.
- Stack: stable `transformers` 5.x, TRL (SFT + GRPO both have a VLM path, not proven for Mistral3), PEFT LoRA on the language-model layers only.
- Metric: fraction of valid JSON, and field-level F1 over (field, value) pairs. One function (`receipts.score`) for every row.
- Rules: choose rank, steps and learning rate on validation. Touch the test split once per row, at the end.

## Pipeline
1. **Before.** Instruct prompted, zero-shot, on test. Also Mistral large via API as the reference row.
2. **SFT** with LoRA (rank 8/16/64 on validation).
3. **After SFT.** Same test eval.
4. **GRPO** on the best SFT adapter. Reward = 0.2 * valid JSON + 0.8 * F1. Then the same test eval.
5. **Speed and cost.** Merge the best adapter, then measure latency per receipt and throughput on the same GPU with HF `generate` and with vLLM
   (if it serves Mistral3). Compare with the large-model API: latency, $ per 1k receipts.
6. **Local run.** See below.

## Day-0 gate (`smoke.py`)
Stages: `data` (CPU, runs on this Mac, passes), `tokens` (processor + chat template, prompt length with image), `sft` (20 steps), `grpo` (5 steps).
The last three need a CUDA GPU. If the stack breaks, fallbacks: GRPO on text-only input, a hand-written GRPO loop, or SFT-only.
TRL's continuous batching raises on multimodal models, so use plain generation or vLLM.

## Local run: what this machine can do
Intel i9, 16 GB, no GPU. PyTorch stops at 2.2.2 here and `transformers` 5 refuses it, so no HF inference locally.
Local would mean a quantized GGUF through llama.cpp on CPU. Whether llama.cpp supports Ministral 3 image input is unverified.
Expect slow image prefill. It is still a real data point: "runs offline on a laptop, X seconds per receipt, Y points of F1 lost to quantization".
Fine-tuning does not make the model faster; speed comes from size and quantization, so the speed rows are 3B vs large API, and GPU vs CPU-quantized.

## Result table (fill in)
| Row | Valid JSON % | Field F1 | s / receipt | $ / 1k receipts |
|---|---|---|---|---|
| Large model, API, prompted | | | | |
| Ministral 3B, prompted (before) | | | | |
| + SFT (best rank) | | | | |
| + SFT + GRPO | | | | |
| SFT merged, vLLM on GPU | | | | |
| SFT merged, Q4 on laptop CPU | | | | |

Report the table even if GRPO adds nothing. Add training and reward curves and 3 failure examples.

## Caveats for the README
- CORD is Indonesian receipts, a generic task, not client data.
- 100 test receipts: report the F1 spread (bootstrap), not just a point value.
- Image resolution is capped at 1024 px.
