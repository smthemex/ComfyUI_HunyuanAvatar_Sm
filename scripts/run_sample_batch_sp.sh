#!/bin/bash
JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./

export MODEL_BASE=./weights
OUTPUT_BASEPATH=./results
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt


torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} 
