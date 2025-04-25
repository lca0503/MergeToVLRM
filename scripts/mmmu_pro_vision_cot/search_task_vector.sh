#!/bin/bash

for lambda in $(seq 1.0 -0.1 0.0); do
    echo "lambda: ${lambda}"
    
    vlseq_model_path="./models_vlseq/mllama_tulu_task_vector_${lambda}/"

    result_path="./results/MMMU_pro_vision_cot/mllama_tulu_task_vector_${lambda}.jsonl"

    python3 src/get_scores.py \
	    --input_path "./best_of_n/mmmu_pro_vision_cot.jsonl" \
	    --output_path ${result_path} \
	    --task "mmmu_pro_vision_cot" \
	    --model_id ${vlseq_model_path}
done

