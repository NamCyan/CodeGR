import os
import torch
import pickle
import logging
import argparse
import jsonlines
import json

import numpy as np

from tqdm import tqdm

from models.ranker import (
    get_ranker_components,
    sentence_to_inputs
)

from utils.model_utils import (
    get_model_obj,
    setup_for_distributed_mode,
    load_states_from_checkpoint
)

from utils.options import (
    setup_args_gpu,
    add_cuda_params,
    add_encoder_params,
    set_encoder_params_from_state
)

class Ranker(object):

    def __init__(self, args):
        self.args = args

        state = load_states_from_checkpoint(args.ranker_file)
        set_encoder_params_from_state(state.encoder_params, self.args, 'ranker')

        self.ranker, _, self.ranker_tokenizer = get_ranker_components(args, True)
        self.ranker, _ = setup_for_distributed_mode(self.ranker, None, args.local_rank, args.fp16, args.fp16_opt_level)

        
        model_to_load = get_model_obj(self.ranker)
        model_to_load.load_state_dict(state.model_dict, strict=False)
    
    def _get_ranker_scores(self, queries, ctx, answer):

        contexts = [ctx for _ in queries]

        inputs = sentence_to_inputs(queries, 
                                    contexts, 
                                    None, 
                                    self.ranker_tokenizer, 
                                    self.args)

        outs = self.ranker(inputs['input_ids'], inputs['attention_mask'])['scores'].cpu().numpy()

        return outs
    
    def _save_file(self, data, file):

        args = self.args

        if args.world_size > 1:
            with open(file + '.%d' % args.local_rank, 'wb') as fw:
                pickle.dump(data, fw)
            torch.distributed.barrier()
            if args.local_rank == 0:
                data = []
                for i in range(args.world_size):
                    with open(file + '.%d' % i, 'rb') as fr:
                        data.extend(pickle.load(fr))
                    os.remove(file + '.%d' % i)

        if args.local_rank in [0, -1]:
            # with jsonlines.open(file, 'w') as fw:
            #     fw.write_all(data)
            with open(file, 'w') as fw:
                for line in data:
                    fw.write(json.dumps(line) + '\n')


    @torch.no_grad()
    def rank(self):
        args = self.args
    
        inputs, outs = [], []
        with jsonlines.open(args.in_file) as reader:
            for obj in reader:
                inputs.append(obj)
        
        if args.world_size > 1:
            size_per_rank = int(len(inputs) / args.world_size)
            start_idx = args.local_rank * size_per_rank
            end_idx = start_idx + size_per_rank if args.local_rank != args.world_size - 1 else len(inputs)
            inputs = inputs[start_idx:end_idx] 
        
        batch_size = 10
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            queries = []
            for item in batch:
                if 'text' in item:
                    queries.append(item['text'])
            
            unique_queries = list(set(queries))
            
            if len(batch) > 0:
                obj = batch[0]  # Using the first object for context
                ctx = dict(text=obj['context']) if 'context' in obj else dict()
                
                if unique_queries:  # Only process if there are queries
                    scores = self._get_ranker_scores(unique_queries, ctx, None)
                    argsort = np.argsort(scores)[::-1]
                    top_k = 5
                    top_indices = argsort[:top_k] if len(argsort) > top_k else argsort
                    top_queries = [unique_queries[i] for i in top_indices]
                    top_scores = scores[top_indices].tolist()
                    query_to_score = dict(zip(top_queries, top_scores))


                    seen_items = set()
                    for item in batch:
                        if 'text' in item and item['text'] in query_to_score and item['text'] not in seen_items:
                            item['score'] = query_to_score[item['text']]
                            outs.append(item)
                            seen_items.add(item['text'])
                        else:
                            continue

                    print(f"Processed {i + len(batch)} / {len(inputs)}")
                else:
                    outs.extend(batch)
            else:
                pass
    
        self._save_file(outs, args.out_file)

def main():
    parser = argparse.ArgumentParser()

    add_cuda_params(parser)
    add_encoder_params(parser)

    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)

    args = parser.parse_args()

    setup_args_gpu(args)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    ranker = Ranker(args)

    try:
        ranker.rank()
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":

    logger = logging.getLogger()

    main()
