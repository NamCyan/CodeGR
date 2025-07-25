from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    RobertaTokenizerFast,
    HfArgumentParser,
    set_seed,
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
import os
set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    lang: str = field(default="Ruby")
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    eval_samples: Optional[int] = field(default=5000)
    test_samples: Optional[int] = field(default=-1)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)


def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        mrr = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                mrr += 1/(hits[0] + 1)
                if hits[0] == 0:
                    hit_at_1 += 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions), "MRR": mrr/len(eval_preds.predictions)}
    return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # # We use wandb logger: https://wandb.ai/site.
    # if training_args.local_rank == 0:  # only on main process
    #     # Initialize wandb run
    #     wandb.login()
    #     wandb.init(project="DSI", name=training_args.run_name)

    # if 'mt5' in run_args.model_name:
    #     tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
    #     fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
    #     if run_args.model_path:
    #         model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
    #     else:
    #         model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    # else:
    tokenizer = AutoTokenizer.from_pretrained(run_args.model_name)
    if "codet5" in run_args.model_name:
        fast_tokenizer = RobertaTokenizerFast.from_pretrained(run_args.model_name)
    else:
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name)
    if run_args.model_path:
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_name)

    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
        )
        trainer.train()

    elif run_args.task == "DSI":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer,
                                             max_samples = run_args.eval_samples)
        
        test_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer,
                                             max_samples = run_args.test_samples)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        if "codet5" in run_args.model_name: 
            SPIECE_UNDERLINE = "Ġ"
        else:
            SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################
        
        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
        if training_args.do_train:
            trainer.train()
        preds, labels, metrics = trainer.predict(test_dataset)
        predictions = tokenizer.batch_decode(preds[:,:,0])
        with open(os.path.join(training_args.output_dir, "predictions.txt"), "w") as f:
            f.write("\n".join(predictions))
        
        print(metrics)
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)


    elif run_args.task == 'generation':
        generate_dataset = GenerateDataset(lang = run_args.lang,
                                           path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        
        def process_query(query):
            return " ".join(query.lower().split())
    
        with open(os.path.join(training_args.output_dir, f"{run_args.lang}.q{run_args.num_return_sequences}"), 'w') as f:
            for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                                desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = tokenizer.decode(tokens, skip_special_tokens=True)
                    jitem = json.dumps({'text_id': docid.item(), 'text': process_query(query)})
                    f.write(jitem + '\n')

    else:
        raise NotImplementedError("--task should be in 'DSI' or 'docTquery' or 'generation'")


if __name__ == "__main__":
    main()

