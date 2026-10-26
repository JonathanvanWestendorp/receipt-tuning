#!/usr/bin/env bash
# Usage: ./vm.sh up | sync | run "cmd" | setup | ssh | pull | start | down
set -e
export CLOUDSDK_ACTIVE_CONFIG_NAME=personal
VM=receipt-tuning
Z=$(gcloud compute instances list --filter="name=$VM" --format='value(zone.basename())' 2>/dev/null)
Z=${Z:-europe-west4-a}

case $1 in
up)
    gcloud compute instances create $VM --zone=$Z --machine-type=g2-standard-8 \
        --accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT --instance-termination-action=STOP --max-run-duration=6h \
        --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
        --boot-disk-size=200GB ;;
sync) gcloud compute scp --zone=$Z receipts.py smoke.py eval.py train.py grpo.py sweep.sh $VM:~ ;;
setup) gcloud compute ssh $VM --zone=$Z --command \
    'nvidia-smi -L && curl -LsSf https://astral.sh/uv/install.sh | sh && ~/.local/bin/uv venv --python 3.12 venv && ~/.local/bin/uv pip install -q --python venv/bin/python torch torchvision "transformers>=5" trl peft datasets pillow mistral-common' ;;
run) gcloud compute ssh $VM --zone=$Z --command "$2" ;;
ssh) gcloud compute ssh $VM --zone=$Z ;;
pull) gcloud compute scp --recurse --zone=$Z "$VM:~/out" "$VM:~/results" . ;;
start) gcloud compute instances start $VM --zone=$Z ;;
down) gcloud compute instances delete $VM --zone=$Z --quiet ;;
*) sed -n 2,3p "$0"; exit 1 ;;
esac
