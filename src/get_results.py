import json
import random
from argparse import ArgumentParser

import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    return args


def extract_jsonl(filename):
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
            
    return data_list


def main(args):
    random.seed(args.seed)
    n_results = extract_jsonl(args.input_path)

    pass_1_cnt = 0
    pass_n_cnt = 0
    random_cnt = 0
    rm_cnt = 0
    for n_result in n_results:
        pass_1_cnt += n_result["correctness"][0]
        pass_n_cnt += max(n_result["correctness"])
        random_cnt += random.choice(n_result["correctness"])
        idx = np.argmax(n_result["scores"])
        rm_cnt += n_result["correctness"][idx]
        
    pass_1_accuracy = pass_1_cnt / len(n_results)
    print(f"Pass @ 1: {pass_1_accuracy * 100}")
    pass_n_accuracy = pass_n_cnt / len(n_results)
    print(f"Pass @ N: {pass_n_accuracy * 100}")
    random_accuracy = random_cnt / len(n_results)
    print(f"Random Accuracy: {random_accuracy * 100}")
    rm_accuracy = rm_cnt / len(n_results)
    print(f"RM Accuracy: {rm_accuracy * 100}")

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
