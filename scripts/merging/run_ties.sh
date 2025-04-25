#!/bin/bash

for lambda in 1.0 0.7 0.5; do
    for density in $(seq 0.8 -0.2 0.2); do
	echo "lambda: ${lambda} density: ${density}"

	config_path="./scripts/merging//config/ties/ties_${lambda}_d_${density}.yaml"

	model_path="./models/mllama_tulu_ties_${lambda}_d_${density}_t/"

	vlseq_model_path="./models_vlseq/mllama_tulu_ties_${lambda}_d_${density}/"
    
	mergekit-yaml ${config_path} ${model_path}

	python3 src/VLRM/map_vlseq.py \
		--tlm_id ${model_path} \
		--seq_id "allenai/llama-3.1-tulu-2-8b-uf-mean-rm" \
		--vlseq_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
		--output_path ${vlseq_model_path}
	
	rm -rf ${model_path}
    done
done
