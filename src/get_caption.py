import os
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, set_seed
from vllm import LLM, SamplingParams

from utils.common import extract_jsonl, save_json
from utils.const import VLMCaptionPromptEnum


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


def captioning(
    task,
    n_results,
    model,
    processor,
    sampling_params,
):
    id_to_images = {}
    if task == "textvqa_val":
        dataset = load_dataset("lmms-lab/textvqa", split="validation")
        for sample in dataset:
            id_to_images[sample["question_id"]] = [sample["image"].convert("RGB")]
    elif task == "mmmu_pro_standard_cot":
        dataset = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
        for sample in dataset:
            id_to_images[sample["id"]] = []
            for i in range(1, 8):
                if sample[f"image_{i}"]:
                    id_to_images[sample["id"]].append(sample[f"image_{i}"])
    elif task == "mmmu_pro_vision_cot":
        dataset = load_dataset("MMMU/MMMU_Pro", "vision", split="test")
        for sample in dataset:
            id_to_images[sample["id"]] = [sample["image"]]
    elif task == "vl_rewardbench":
        dataset = load_dataset("MMInstruction/VL-RewardBench", split="test")
        for sample in dataset:
            id_to_images[sample["id"]] = [sample["image"]]
    else:
        raise NotImplementedError("Task name not found")

    id_to_caption = {}
    for n_result in tqdm(n_results):
        question_id = n_result["id"]
        images = id_to_images[question_id]
        query = n_result["input"]

        user_prompt = VLMCaptionPromptEnum.user_prompt.format(
            INSTRUCTION=query,
        )
        message = [
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_prompt}
             ]
            }
        ]
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }
        ]
        outputs = model.generate(inputs, sampling_params)
        caption = outputs[0].outputs[0].text
        id_to_caption[question_id] = caption

    return id_to_caption


def main(args):
    set_seed(args.seed)

    print(f"{args.input_path} Extracting ...")

    if args.task == "vl_rewardbench":
        dataset = load_dataset("MMInstruction/VL-RewardBench", split="test")
        n_results = []
        for sample in dataset:
            n_results.append(
                {
                    "id": sample["id"],
                    "input": sample["query"]
                }
            )
    else:
        n_results = extract_jsonl(args.input_path)

    print(f"{args.model_id} Initializing ...")

    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    mm = 8 if args.task == "mmmu_pro_standard_cot" else 1
    model = LLM(
        model=args.model_id,
        max_model_len=8192,
        max_num_seqs=1,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        dtype=torch.float16,
        seed=args.seed,
        limit_mm_per_prompt={"image": mm},
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        seed=args.seed,
    )

    id_to_caption = captioning(
        task=args.task,
        n_results=n_results,
        model=model,
        processor=processor,
        sampling_params=sampling_params,
    )

    print(f"Saving results to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_json(args.output_path, id_to_caption)
    print("Results saved successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
