#!/bin/bash

root_dir="$1"

for i in $(seq 0 1 7); do
    python3 -m accelerate.commands.launch \
	    -m lmms_eval \
	    --model llama_vision \
	    --model_args pretrained=meta-llama/Llama-3.2-11B-Vision-Instruct,dtype=float16,device_map=auto \
	    --tasks textvqa_train_100 \
	    --gen_kwargs temperature=1.0,do_sample=True,top_p=1.0 \
	    --batch_size 1 \
	    --log_samples \
	    --seed ${i} \
	    --output_path ${root_dir}/textvqa_train_100/seed_${i}/
done
