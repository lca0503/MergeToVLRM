import json
import os


def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)

        
def save_jsonl(filename, ds):
    with open(filename, 'w') as f:
        for d in ds:
            f.write(json.dumps(d) + '\n')


def extract_json(filename):
    with open(filename, "r") as f:
        data_list = json.load(f)

    return data_list


def extract_jsonl(filename):
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
            
    return data_list


def get_all_jsonl_files(directory):
    jsonl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files
