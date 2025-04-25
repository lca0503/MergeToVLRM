#!/bin/bash

for lambda in $(seq 1.0 -0.1 0.0); do
    echo "lambda: ${lambda}"

    config_path="./scripts/merging/config/task_vector/task_vector_${lambda}.yaml"

    model_path="./models/mllama_tulu_task_vector_${lambda}_t/"

    vlseq_model_path="./models_vlseq/mllama_tulu_task_vector_${lambda}/"
    
    mergekit-yaml ${config_path} ${model_path}

    python3 src/VLRM/map_vlseq.py \
	    --tlm_id ${model_path} \
	    --seq_id "allenai/llama-3.1-tulu-2-8b-uf-mean-rm" \
	    --vlseq_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
	    --output_path ${vlseq_model_path}

    rm -rf ${model_path}
done
