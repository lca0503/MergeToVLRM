import os
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoProcessor,
                          AutoTokenizer, set_seed)

from utils.common import save_jsonl
from utils.prompt import format_prompts, format_text_only_prompts, format_no_image_prompts
from utils.scoring import scoring, text_only_scoring, no_image_scoring
from VLRM.modeling_mllama_cls import MllamaForSequenceClassification


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--no_image", action="store_true")
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    return args


def evaluate(dataset, model, processor, tokenizer, text_only, no_image):
    correct_predictions = 0
    results = []
    for sample in tqdm(dataset):
        query = sample["question"]
        image = sample["image"]
        responses = [sample["chosen"], sample["rejected"]]

        if text_only:
            prompts = format_text_only_prompts(
                query, responses, tokenizer
            )
            outputs = text_only_scoring(
                prompts, model, tokenizer
            )
            predicted_answer = int(np.argmax(outputs))
        elif no_image:
            prompts = format_no_image_prompts(
                query, responses, processor
            )
            outputs = no_image_scoring(
                prompts, model, processor
            )
            predicted_answer = int(np.argmax(outputs))
        else:
            prompts = format_prompts(
                query, responses, processor
            )
            outputs = scoring(
                prompts, image, model, processor
            )
            predicted_answer = int(np.argmax(outputs))
        
        results.append(
            {
                "prompts": prompts,
                "outputs": outputs,
                "predicted_answer": predicted_answer,
                "ground_truth": [0, 1], 
            }
        )

        if predicted_answer == 0:
            correct_predictions += 1

    overall_accuracy = correct_predictions / len(dataset)
    print(f"Overall accuracy: {overall_accuracy * 100}")

    return results


def main(args):
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading dataset ...")
    
    dataset = load_dataset(
        "lca0503/rlaif_v_train_400",
        split="test"
    )
    
    print(f"{args.model_id} Initializing ...")

    processor = None
    tokenizer = None
    if args.text_only:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    else:
        model = MllamaForSequenceClassification.from_pretrained(
            args.model_id,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_id)

    results = evaluate(
        dataset=dataset,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        text_only=args.text_only,
        no_image=args.no_image
    )

    print(f"Saving results to {args.output_path}...")
    save_jsonl(args.output_path, results)
    print("Results saved successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
