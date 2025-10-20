import json
import jsonlines
import argparse
from tqdm import tqdm

def merge_files(file_a, file_b, output_file):
    print(f"Loading data from {file_a}...")
    data_a = []
    with jsonlines.open(file_a) as reader:
        for item in reader:
            data_a.append(item)
    
    print(f"Loading data from {file_b}...")
    # Create a dictionary for faster lookup by text_id
    data_b_dict = {}
    with jsonlines.open(file_b) as reader:
        for item in reader:
            data_b_dict[item["text_id"]] = item
    
    print("Merging data...")
    matches = 0
    for item_a in tqdm(data_a):
        if "text_id" in item_a and str(item_a["text_id"]) in data_b_dict:
            item_b = data_b_dict[str(item_a["text_id"])]
            if "text" in item_b and item_b["text"].startswith("Code"):
                item_a["context"] = item_b["text"]
                matches += 1
    
    print(f"Found and updated {matches} matching items")
    
    print(f"Writing merged data to {output_file}...")
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(data_a)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge data from two JSONL files based on text_id")
    parser.add_argument("--file_a", type=str, default="/root/workspace/CodeGR/DSI-QG/process_data/augmentation/Ruby.q10")
    parser.add_argument("--file_b", type=str, default="/root/workspace/CodeGR/DSI-QG/process_data/Ruby_train_r1.0.json")
    parser.add_argument("--output", type=str, default="/root/workspace/CodeGR/DSI-QG/process_data/augmentation/Ruby.q10.updated.jsonl")
    
    args = parser.parse_args()
    
    merge_files(args.file_a, args.file_b, args.output)