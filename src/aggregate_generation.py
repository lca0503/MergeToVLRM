import os
from argparse import ArgumentParser
from collections import defaultdict

from utils.common import extract_jsonl, get_all_jsonl_files, save_jsonl


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()

    return args


def aggregate_textvqa_results(jsonl_files):
    aggregated = defaultdict(lambda: {
        "id": None,
        "question": None,
        "answers": None,
        "input": None,
        "responses": [],
        "correctness": []
    })

    for jsonl_file in jsonl_files:
        results = extract_jsonl(jsonl_file)
        for result in results:
            assert len(result["resps"]) == 1 and len(result["resps"][0]) == 1
            key = (result["doc"]["question_id"], result["doc"]["question"],
                   tuple(result["doc"]["answers"]), result["input"])
            aggregated_entry = aggregated[key]
            if aggregated_entry["id"] is None:
                aggregated_entry["id"] = result["doc"]["question_id"]
                aggregated_entry["question"] = result["doc"]["question"]
                aggregated_entry["answers"] = result["doc"]["answers"]
                aggregated_entry["input"] = result["input"]
            aggregated_entry["responses"].append(result["resps"][0][0])
            aggregated_entry["correctness"].append(result["exact_match"])

    return list(aggregated.values())


def aggregate_mmmu_pro_standard_results(jsonl_files):
    aggregated = defaultdict(lambda: {
        "id": None,
        "subject": None,
        "question": None,
        "answer": None,
        "input": None,
        "responses": [],
        "correctness": []
    })

    for jsonl_file in jsonl_files:
        results = extract_jsonl(jsonl_file)
        for result in results:
            assert len(result["resps"]) == 1 and len(result["resps"][0]) == 1
            key = (result["doc"]["id"], result["doc"]["subject"], result["doc"]["question"],
                   result["doc"]["answer"], result["input"])
            aggregated_entry = aggregated[key]
            if aggregated_entry["id"] is None:
                aggregated_entry["id"] = result["doc"]["id"]
                aggregated_entry["subject"] = result["doc"]["subject"]
                aggregated_entry["question"] = result["doc"]["question"]
                aggregated_entry["answer"] = result["doc"]["answer"]
                aggregated_entry["input"] = result["input"]
            aggregated_entry["responses"].append(result["resps"][0][0])
            correctness = (result["mmmu_acc"]["answer"] == result["mmmu_acc"]["parsed_pred"])
            aggregated_entry["correctness"].append(correctness)

    return list(aggregated.values())


def aggregate_mmmu_pro_vision_results(jsonl_files):
    aggregated = defaultdict(lambda: {
        "id": None,
        "subject": None,
        "answer": None,
        "input": None,
        "responses": [],
        "correctness": []
    })

    for jsonl_file in jsonl_files:
        results = extract_jsonl(jsonl_file)
        for result in results:
            assert len(result["resps"]) == 1 and len(result["resps"][0]) == 1
            key = (result["doc"]["id"], result["doc"]["subject"],
                   result["doc"]["answer"], result["input"])
            aggregated_entry = aggregated[key]
            if aggregated_entry["id"] is None:
                aggregated_entry["id"] = result["doc"]["id"]
                aggregated_entry["subject"] = result["doc"]["subject"]
                aggregated_entry["answer"] = result["doc"]["answer"]
                aggregated_entry["input"] = result["input"]
            aggregated_entry["responses"].append(result["resps"][0][0])
            correctness = (result["mmmu_acc"]["answer"] == result["mmmu_acc"]["parsed_pred"])
            aggregated_entry["correctness"].append(correctness)

    return list(aggregated.values())


def main(args):
    jsonl_files = get_all_jsonl_files(args.input_dir)

    if args.task == "textvqa_val" or args.task == "textvqa_train_100":
        aggregated_results = aggregate_textvqa_results(jsonl_files)
    elif args.task == "mmmu_pro_standard_cot":
        aggregated_results = aggregate_mmmu_pro_standard_results(jsonl_files)
    elif args.task == "mmmu_pro_vision_cot":
        aggregated_results = aggregate_mmmu_pro_vision_results(jsonl_files)
    else:
        raise NotImplementedError("Task name not found")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_jsonl(args.output_path, aggregated_results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
