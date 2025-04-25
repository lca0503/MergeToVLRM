import argparse
import csv
import json
from collections import Counter


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate accuracies from JSONL file.")
    parser.add_argument(
        "--input_path", type=str, required=True, 
        help="Path to the input JSONL file."
    )
    
    return parser.parse_args()


def process_file(input_path):
    """
    Process the JSONL file and calculate counts and accuracies for each category.
    Args:
        input_path (str): Path to the input JSONL file.
    Returns:
        dict: A dictionary with accuracies and counts for each category.
    """
    hallucination_cnt = 0
    reasoning_cnt = 0
    general_cnt = 0
    ids = []
    total_records = 0

    with open(input_path, "r") as file:
        for line in file:
            record = json.loads(line)
            total_records += 1
            idx = record["id"]
            is_correct = record["predicted_answer"] == record["ground_truth"][0]

            if "mmmu_pro" in idx or "mathverse" in idx:
                ids.append("reasoning")
                reasoning_cnt += is_correct
            elif "hallucination_pair" in idx or "RLAIF-V" in idx or "RLHF-V" in idx:
                ids.append("hallucination")
                hallucination_cnt += is_correct
            else:
                ids.append("general")
                general_cnt += is_correct

    # Calculate total correct
    total_correct = reasoning_cnt + hallucination_cnt + general_cnt

    counts = Counter(ids)
    results = {
        "reasoning": {
            "count": counts["reasoning"],
            "accuracy": reasoning_cnt / counts["reasoning"]
        },
        "hallucination": {
            "count": counts["hallucination"],
            "accuracy": hallucination_cnt / counts["hallucination"]
        },
        "general": {
            "count": counts["general"],
            "accuracy": general_cnt / counts["general"]
        },
        "overall_accuracy": total_correct / total_records,
        "macro_average_accuracy": (
            sum([
                reasoning_cnt / counts["reasoning"],
                hallucination_cnt / counts["hallucination"],
                general_cnt / counts["general"]
            ]) / 3
        )
    }

    return results


def main():
    """Main function to parse arguments and compute results."""
    args = parse_arguments()
    print(args.input_path)
    results = process_file(args.input_path)
    print(results)

    
if __name__ == "__main__":
    main()
