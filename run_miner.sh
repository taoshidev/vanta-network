#!/bin/bash
cd /home/rizzo/vanta-network
export PYTHONPATH="/home/rizzo/vanta-network:$PYTHONPATH"
exec /home/rizzo/miniconda3/envs/vanta/bin/python neurons/miner.py \
    --wallet.name local_miner \
    --wallet.hotkey default \
    --netuid 2 \
    --subtensor.network ws://127.0.0.1:9945 \
    --subtensor.chain_endpoint ws://127.0.0.1:9945
