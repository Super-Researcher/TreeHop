import os
import json
import jsonlines
from enum import Enum


class NodeType(Enum):
    query = -1
    irrelevant_doc = 0
    relevant_doc = 1
    leaf = 2


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file_jsonl(input_file_path: str):
    if input_file_path.endswith(".json"):
        with open(input_file_path) as f:
            input_data = json.load(f)
    else:
        input_data = load_jsonlines(input_file_path)
    return input_data


def save_file_jsonl(data, output_file_path, mode='w'):
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    if output_file_path.endswith("json"):
        with open(output_file_path) as f:
            json.dump(data, f)
    else:
        with jsonlines.open(output_file_path, mode=mode) as f:
            f.write_all(data)
