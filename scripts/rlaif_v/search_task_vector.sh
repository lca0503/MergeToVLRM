#!/bin/bash

for lambda in $(seq 1.0 -0.1 0.0); do
    echo "lambda: ${lambda}"

    vlseq_model_path="./models_vlseq/mllama_tulu_task_vector_${lambda}/"

    result_path="./results/RLAIF_V/mllama_tulu_task_vector_${lambda}.jsonl"

    python3 src/rlaif_v.py \
            --output_path ${result_path} \
            --model_id ${vlseq_model_path}
done
