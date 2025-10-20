import json
import argparse
from collections import defaultdict

def merge_queries(file_a, file_b, output_file):
    # Read original queries (file A)
    original_queries = {}
    with open(file_a, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            text_id = item["text_id"]
            if item["text"].startswith("Query:"):
                query = item["text"]
                original_queries[text_id] = query[6:].strip()  # Remove "Query: " prefix
    
    print(f"Loaded {len(original_queries)} original queries from {file_a}")
    
    # Read augmented queries (file B)
    augmented_queries = defaultdict(list)
    with open(file_b, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            text_id = str(item["text_id"])
            query = item["text"]
            augmented_queries[text_id].append(query)
    
    print(f"Loaded augmented queries for {len(augmented_queries)} documents from {file_b}")
    
    # Merge queries
    merged_results = []
    total_original_added = 0
    
    for text_id, aug_queries in augmented_queries.items():
        if text_id in original_queries:
            is_unique = True
            
            for aug_query in aug_queries:
                if original_queries[text_id].lower().strip() == aug_query.lower().strip():
                    is_unique = False
                merged_results.append({
                    "text_id": text_id,
                    "text": aug_query,
                    "is_original": False
                })
            
            # Add original query if unique
            if is_unique:
                merged_results.append({
                    "text_id": text_id,
                    "text": original_queries[text_id],
                    "is_original": True
                })
                total_original_added += 1
        else:
            for aug_query in aug_queries:
                merged_results.append({
                    "text_id": text_id,
                    "text": aug_query,
                    "is_original": False
                })
            
    
    # Write merged results
    with open(output_file, 'w') as f:
        for item in merged_results:
            # Remove the is_original field before writing
            output_item = {"text_id": item["text_id"], "text": item["text"], "is_original": item["is_original"]}
            f.write(json.dumps(output_item) + '\n')
    
    print(f"Merged results written to {output_file}")
    print(f"Added {total_original_added} original queries that were unique")
    print(f"Total queries in output: {len(merged_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge original queries with augmented queries")
    parser.add_argument("--file_a", type=str,  default="/root/workspace/CodeGR/data/original_indexed_data/Ruby_train_r1.0.json", help="Path to file with original queries")
    parser.add_argument("--file_b", type=str,  default="/root/workspace/CodeGR/data/augmentation/Ruby.jsonl", help="Path to file with augmented queries")
    parser.add_argument("--output", type=str,  default="/root/workspace/CodeGR/data/augmentation/Ruby_ready_to_feed.jsonl", help="Path to output file")

    args = parser.parse_args()
    
    merge_queries(args.file_a, args.file_b, args.output)