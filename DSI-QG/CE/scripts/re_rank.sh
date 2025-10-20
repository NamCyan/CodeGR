for lang in Ruby
do
    python3 re-rank.py  \
    --in_file /root/workspace/CodeGR/DSI-QG/process_data/augmentation/Ruby.q10.updated.jsonl \
    --out_file /root/workspace/CodeGR/DSI-QG/process_data/augmentation/Ruby.jsonl \
    --ranker_file /root/workspace/CodeGR/DSI-QG/CE/Ranker/ranker.pt \
    --no_title
done