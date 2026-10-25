"""Shared bits: CORD-v2 -> (prompt, image, target) and the reward/metric used for every table row."""
import json
import re
from collections import Counter

from datasets import load_dataset

MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"  # Base-2512 is the optional ablation
PROMPT = "Extract this receipt as JSON with keys menu (list of nm, cnt, price), sub_total and total. Output JSON only."
SCHEMA_PROMPT = PROMPT + """ Use exactly CORD's keys and copy every value as a string, as printed on the receipt.
menu: list of items with nm (name), cnt (quantity), price, and where printed unitprice, num, discountprice.
sub_total: subtotal_price, tax_price, service_price, discount_price, etc.
total: total_price, cashprice, changeprice, creditcardprice, emoneyprice, menuqty_cnt.
Example: {"menu": [{"nm": "ICE TEA", "cnt": "2", "price": "20,000"}], "sub_total": {"subtotal_price": "20,000", "tax_price": "2,000"}, "total": {"total_price": "22,000", "cashprice": "25,000", "changeprice": "3,000"}}"""
MAX_SIDE = 1024  # caps image tokens; a few CORD images are 2304x4096


def load(split, n=None, prompt=PROMPT):
    ds = load_dataset("naver-clova-ix/cord-v2", split=split)
    if n:
        ds = ds.select(range(n))

    def fmt(ex):
        img = ex["image"].convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        gt = json.loads(ex["ground_truth"])["gt_parse"]
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)}],
            "image": img,
            "gt": json.dumps(gt, ensure_ascii=False),
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def parse(text):
    """Model output -> dict, or None if it is not a JSON object."""
    m = re.search(r"\{.*\}", text, re.S)  # tolerates ```json fences and chatter
    try:
        out = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def flatten(x, path=""):
    # menu is a list for most receipts but a bare dict for one-item ones; this treats both the same
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in flatten(v, f"{path}.{k}" if path else k)]
    if isinstance(x, list):
        return [p for v in x for p in flatten(v, path)]
    return [(path, str(x))]


def f1(pred, gt):
    # ponytail: multiset of (field, value) pairs ignores item order; swap for Hungarian matching if that matters
    a, b = Counter(flatten(pred)), Counter(flatten(gt))
    tp = sum((a & b).values())
    if not tp:
        return 0.0
    p, r = tp / sum(a.values()), tp / sum(b.values())
    return 2 * p * r / (p + r)


def score(text, gt):
    """(valid_json, f1) for one completion string and a gt JSON string."""
    pred = parse(text)
    return (0.0, 0.0) if pred is None else (1.0, f1(pred, json.loads(gt)))


def reward(completions, gt, **kwargs):
    # GRPO passes conversational completions as [{"role": "assistant", "content": str}]
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [0.2 * v + 0.8 * f for v, f in (score(t, g) for t, g in zip(texts, gt))]


if __name__ == "__main__":
    g = json.dumps({"menu": {"nm": "Tea", "price": "1"}, "total": {"total_price": "1"}})
    assert reward(["```json\n" + g + "\n```"], [g]) == [1.0]
    assert reward(["no json here"], [g]) == [0.0]
    assert 0 < reward(['{"total": {"total_price": "1"}}'], [g])[0] < 1
    print("receipts.py ok")
