import argparse
import csv
import json
from collections import Counter


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate accuracy from JSONL file.")
    parser.add_argument("--input_path", type=str, required=True)
    
    return parser.parse_args()


def process_file(input_path):
    """
    Process the JSONL file and calculate counts and accuracies for each category.
    Args:
        input_path (str): Path to the input JSONL file.
    Returns:
        dict: A dictionary with accuracies and counts for each category.
    """
    is_correct = 0
    total_records = 0
    
    with open(input_path, "r") as file:
        for line in file:
            record = json.loads(line)
            total_records += 1
            is_correct += (record["predicted_answer"] == record["ground_truth"][0])

    print(is_correct * 100 / total_records)

    
def main():
    """Main function to parse arguments and compute results."""
    args = parse_arguments()
    print(args.input_path) 
    process_file(args.input_path)
   
    
if __name__ == "__main__":
    main()
