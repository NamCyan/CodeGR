#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Create directory for outputs
mkdir -p /root/workspace/CodeGR/data/ranker_data

for LANG in Ruby 
do
  echo "Processing $LANG data..."
  python fuel4ranker.py \
    --data_path /root/workspace/CodeGR/data/original_indexed_data/ \
    --save_dir /root/workspace/CodeGR/data/ranker_data \
    --tokenizer Salesforce/codet5p-220m \
    --split $LANG \
    --num_hard_negatives 3
done
