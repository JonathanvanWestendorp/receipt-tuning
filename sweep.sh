#!/usr/bin/env bash
# Runs on the VM, skips finished steps so it can be restarted after a preemption.
cd ~ && PY=venv/bin/python
[ -f results/base_schema_prompted.json ] || SCHEMA=1 $PY eval.py base_schema_prompted            # test, once
[ -f results/val_base_prompted.json ] || SPLIT=validation $PY eval.py val_base_prompted
for r in 8 16 64; do
    [ -f out/sft_r$r/adapter_config.json ] || $PY train.py $r
    [ -f results/val_sft_r$r.json ] || SPLIT=validation $PY eval.py val_sft_r$r out/sft_r$r
done
echo SWEEP DONE
