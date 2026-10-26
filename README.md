# Receipt extraction with Ministral 3B

A small experiment: can a 3B vision model learn to read receipts and return clean JSON? Ministral 3 3B Instruct is asked to extract a receipt as JSON, then fine-tuned with LoRA, and the two are compared. Data is CORD-v2 (800 train, 100 validation, 100 test receipts). Training ran on one L4 GPU.

| Model | Valid JSON | Field F1 | 95% CI | Seconds per receipt |
|---|---|---|---|---|
| Prompted, short prompt | 95% | 0.212 | 0.18 to 0.25 | 5.3 |
| Prompted, prompt lists the CORD keys | 100% | 0.588 | 0.54 to 0.64 | 6.4 |
| Fine-tuned (LoRA rank 8) | 100% | 0.871 | 0.84 to 0.90 | 5.8 |
| Fine-tuned, then GRPO (300 steps) | 100% | 0.857 | 0.82 to 0.89 | 5.6 |

Test split, 100 receipts, greedy decoding. Field F1 compares the (field, value) pairs in the output with the labels. The interval is a bootstrap over receipts. The fair comparison is the second row, since the first row never saw the key names.

Notes:
- LoRA ranks 8, 16 and 64 scored 0.900, 0.887 and 0.888 on the validation split, which is the same within noise. Rank 8 was used.
- The key list in the second prompt was taken from the training labels, so that baseline knows the label vocabulary.
- About a quarter of the remaining errors look like label formatting (separators, stray punctuation). The rest are misread values and fields the labels have or skip.
- GRPO did not help. Against the fine-tuned model, test F1 changed by -0.014 (paired 95% interval -0.036 to +0.005) and validation F1 by -0.007. One receipt with 8 sampled answers was used per step, and a third of the steps had identical rewards for all 8 answers, so they gave no learning signal. The fine-tuned model already returns valid JSON, so the reward had little left to improve.
- CORD is Indonesian receipts.
