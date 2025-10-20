from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="datadrive5/namlh35/CodeGR/data/dsi_data")
parser.add_argument('--save_dir', type=str, default="datadrive5/namlh35/CodeGR/data/dsi_data")
parser.add_argument('--tokenizer', type=str, default="microsoft/unixcoder-base")
parser.add_argument('--split', type=str, default="Ruby")
parser.add_argument('--num_hard_negatives', type=int, default=3)
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

train_file = os.path.join(args.data_path, f"{args.split}_train_r1.0.json")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

full_dataset = load_dataset("json", data_files=train_file, split="train")

code_dataset = full_dataset.filter(
    lambda example: example["text"].startswith("Code:")
).map(lambda example: {"text": example["text"][5:].strip()})

query_dataset = full_dataset.filter(
    lambda example: example["text"].startswith("Query:")
).map(lambda example: {"text": example["text"][6:].strip()})

code_dataset = code_dataset.map(lambda example: {"tokenized_text": tokenizer.tokenize(example["text"])})
tokenized_corpus = code_dataset["tokenized_text"]
documents = code_dataset["text"]
corpus_ids = code_dataset["text_id"]

query_dataset = query_dataset.map(lambda example: {"tokenized_text": tokenizer.tokenize(example["text"])})
queries = query_dataset["tokenized_text"]
query_texts = query_dataset["text"]
query_ids = query_dataset["text_id"]

print(f"{args.split}: {len(documents)} documents and {len(queries)} queries loaded")

bm25 = BM25Okapi(tokenized_corpus)

def get_data(arg):
    query_idx, query_id = arg
    query = queries[query_idx]
    query_text = query_texts[query_idx]
    
    try:
        positive_idx = corpus_ids.index(query_id)
        positive_doc = documents[positive_idx]
    except ValueError:
        return None
    
    doc_scores = bm25.get_scores(query)
    
    k = args.num_hard_negatives + 1
    top_indices = np.argsort(doc_scores)[-k*2:][::-1].tolist()
    
    hard_negatives = []
    for idx in top_indices:
        if idx != positive_idx and len(hard_negatives) < args.num_hard_negatives:
            hard_negatives.append({"text": documents[idx]})
    
    data = {
        "question": query_text,
        "answers": [],  # Empty as per schema
        "positive_ctxs": [{"text": positive_doc}],
        "negative_ctxs": [],  # Empty as per schema
        "hard_negative_ctxs": hard_negatives
    }

    return data

args_list = [(i, qid) for i, qid in enumerate(query_ids)]

# results = []
# with ProcessPoolExecutor(max_workers=2) as executor:
#     futures = [executor.submit(get_data, arg) for arg in args_list]
#     for f in tqdm(as_completed(futures), total=len(futures)):
#         result = f.result()
#         if result:  
#             results.append(result)
args_list = [(i, qid) for i, qid in enumerate(query_ids)]

results = []
for arg in tqdm(args_list, desc="Processing queries"):
    result = get_data(arg)
    if result:
        results.append(result)

output_file = os.path.join(args.save_dir, f"{args.split}_ranker_data.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Created dataset with {len(results)} examples at {output_file}")