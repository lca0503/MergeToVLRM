#!/bin/bash

for lambda in 1.0 0.7 0.5; do
    for density in $(seq 0.8 -0.2 0.2); do
	echo "lambda: ${lambda} density: ${density}"

	vlseq_model_path="./models_vlseq/mllama_tulu_dare_ties_${lambda}_d_${density}/"

	result_path="./results/MMMU_pro_vision_cot/mllama_tulu_dare_ties_${lambda}_d_${density}.jsonl"
	
	python3 src/get_scores.py \
		--input_path "./best_of_n/mmmu_pro_vision_cot.jsonl" \
		--output_path ${result_path} \
		--task "mmmu_pro_vision_cot" \
		--model_id ${vlseq_model_path}
    done
done

