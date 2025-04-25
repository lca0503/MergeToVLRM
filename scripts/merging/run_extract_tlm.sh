#!/bin/bash

python3 src/VLRM/extract_tlm.py \
	--tlm_id "meta-llama/Llama-3.1-8B-Instruct" \
	--vlm_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
	--output_path "./models/mllama_t" \
	--source "vlm"

python3 src/VLRM/extract_tlm.py \
	--tlm_id "meta-llama/Llama-3.1-8B-Instruct" \
	--seq_id "allenai/llama-3.1-tulu-2-8b-uf-mean-rm" \
	--output_path "./models/tulu_t" \
	--source "seq"
