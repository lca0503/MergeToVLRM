#!/bin/bash

for lambda in 1.0 0.7 0.5; do
    for density in $(seq 0.8 -0.2 0.2); do
	echo "lambda: ${lambda} density: ${density}"

	vlseq_model_path="./models_vlseq/mllama_tulu_ties_${lambda}_d_${density}/"

	result_path="./results/VL_RewardBench/mllama_tulu_ties_${lambda}_d_${density}.jsonl"

	python3 src/vl_rewardbench.py \
		--output_path ${result_path} \
		--model_id ${vlseq_model_path}
    done
done

