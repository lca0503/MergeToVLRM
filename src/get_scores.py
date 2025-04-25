import copy
import os
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoProcessor,
                          AutoTokenizer, MllamaForConditionalGeneration,
                          set_seed)

from utils.common import extract_json, extract_jsonl, save_jsonl
from utils.prompt import (format_prompts, format_text_only_prompts,
                          format_vlm_only_prompts, format_no_image_prompts)
from utils.scoring import scoring, text_only_scoring, vlm_only_scoring, no_image_scoring
from VLRM.modeling_mllama_cls import MllamaForSequenceClassification


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--vlm_only", action="store_true")
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--caption_path", type=str, default="")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


def evaluate(
    task,
    n_results,
    model,
    processor,
    tokenizer,
    sampling_params,
    text_only,
    vlm_only,
    no_image,
    caption,
    caption_path,
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
    else:
        raise NotImplementedError("Task name not found")

    if caption:
        id_to_caption = extract_json(caption_path)

    total_correctness = 0
    scores_results = []
    for n_result in tqdm(n_results):
        question_id = n_result["id"]
        images = id_to_images[question_id]
        responses = n_result["responses"]
        query = n_result["input"]

        scores_result = copy.deepcopy(n_result)

        if text_only:
            prompts = format_text_only_prompts(
                query, responses, tokenizer
            )
            scores = text_only_scoring(
                prompts, model, tokenizer
            )
            scores_result["scores"] = scores
        elif caption:
            query = id_to_caption[str(question_id)] + " " + query
            prompts = format_text_only_prompts(
                query, responses, tokenizer
            )
            scores = text_only_scoring(
                prompts, model, tokenizer
            )
            scores_result["scores"] = scores
        elif vlm_only:
            prompts = format_vlm_only_prompts(
                query, responses, processor, len(images)
            )
            vlm_outputs, scores = vlm_only_scoring(
                prompts, images, model, sampling_params
            )
            scores_result["vlm_outputs"] = vlm_outputs
            scores_result["scores"] = scores
        elif no_image:
            prompts = format_no_image_prompts(
                query, responses, processor
            )
            scores = no_image_scoring(
                prompts, model, processor
            )
            scores_result["scores"] = scores
        else:
            prompts = format_prompts(
                query, responses, processor, len(images)
            )
            scores = scoring(
                prompts, images, model, processor
            )
            scores_result["scores"] = scores
        scores_results.append(scores_result)

        predicted_answer = int(np.argmax(scores))
        total_correctness += scores_result["correctness"][predicted_answer]

    overall_accuracy = total_correctness / len(scores_results)
    print(f"Overall accuracy: {overall_accuracy * 100}")

    return scores_results


def main(args):
    set_seed(args.seed)

    print(f"{args.input_path} Extracting ...")

    n_results = extract_jsonl(args.input_path)

    print(f"{args.model_id} Initializing ...")

    processor = None
    tokenizer = None
    sampling_params = None
    if args.text_only or args.caption:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    elif args.vlm_only:
        os.environ["CUDA_LAUNCH_BLOCKING"]="1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm import LLM, SamplingParams
        mm = 8 if args.task == "mmmu_pro_standard_cot" else 1
        model = LLM(
            model=args.model_id,
            max_model_len=8192,
            max_num_seqs=4,
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
    else:
        model = MllamaForSequenceClassification.from_pretrained(
            args.model_id,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_id)

    results = evaluate(
        task=args.task,
        n_results=n_results,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        text_only=args.text_only,
        vlm_only=args.vlm_only,
        no_image=args.no_image,
        caption=args.caption,
        caption_path=args.caption_path,
    )

    print(f"Saving results to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_jsonl(args.output_path, results)
    print("Results saved successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
