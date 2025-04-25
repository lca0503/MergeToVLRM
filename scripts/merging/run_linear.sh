#!/bin/bash

for rm_ratio in $(seq 1.0 -0.1 0.0); do
    ratio=$(echo "1 - ${rm_ratio}" | bc);
    lm_ratio=$(printf "%.1f" ${ratio});
    echo "rm_ratio: ${rm_ratio} lm_ratio: ${lm_ratio}"

    config_path="./scripts/merging/config/linear/linear_${lm_ratio}_${rm_ratio}.yaml"

    model_path="./models/mllama_tulu_linear_${lm_ratio}_${rm_ratio}_t/"

    vlseq_model_path="./models_vlseq/mllama_tulu_linear_${lm_ratio}_${rm_ratio}/"
    
    mergekit-yaml ${config_path} ${model_path}

    python3 src/VLRM/map_vlseq.py \
	    --tlm_id ${model_path} \
	    --seq_id "allenai/llama-3.1-tulu-2-8b-uf-mean-rm" \
	    --vlseq_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
	    --output_path ${vlseq_model_path}

    rm -rf ${model_path}
done
