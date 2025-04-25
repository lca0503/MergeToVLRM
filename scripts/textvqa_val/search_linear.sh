#!/bin/bash

for rm_ratio in $(seq 1.0 -0.1 0.0); do
    ratio=$(echo "1 - ${rm_ratio}" | bc);
    lm_ratio=$(printf "%.1f" ${ratio});
    echo "rm_ratio: ${rm_ratio} lm_ratio: ${lm_ratio}"
    
    vlseq_model_path="./models_vlseq/mllama_tulu_linear_${lm_ratio}_${rm_ratio}/"

    result_path="./results/TextVQA_val/mllama_tulu_linear_${lm_ratio}_${rm_ratio}.jsonl"

    python3 src/get_scores.py \
	    --input_path "./best_of_n/textvqa_val.jsonl" \
	    --output_path ${result_path} \
	    --task "textvqa_val" \
	    --model_id ${vlseq_model_path}
done

