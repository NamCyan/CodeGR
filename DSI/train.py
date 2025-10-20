from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import AutoTokenizer, T5TokenizerFast, T5ForConditionalGeneration, TrainingArguments, TrainerCallback # T5Tokenizer, AutoTokenizer
from trainer import IndexingTrainer
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging, argparse
import json, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: AutoTokenizer): # tokenizer: T5Tokenizer AutoTokenizer
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        mrr = 0
        model = kwargs['model'].eval()
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=self.args.id_max_length,
                    num_beams=20,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping =True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        mrr += 1/(hits[0] + 1)
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.info({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@10": hit_at_10 / len(self.test_dataset), "mrr": mrr / len(self.test_dataset)})


def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label = np.where(label != -100, label, tokenizer.pad_token_id)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Salesforce/codet5-base")
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--id_max_length', type=int, default=20)
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_samples', type=int, default=5000)

    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model_name = args.model_name
    L = args.max_length #32  # only use the first 32 tokens of documents (including title)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_dataset = IndexingTrainDataset(path_to_data=args.train_data,
                                         max_length=L,
                                         tokenizer=tokenizer,
                                         cache_dir=args.cache_dir,
                                         max_samples=args.train_samples)
    
    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    eval_dataset = IndexingTrainDataset(path_to_data=args.train_data,
                                        max_length=L,
                                        tokenizer=tokenizer,
                                        cache_dir=args.cache_dir,
                                        max_samples=args.eval_samples)
    
    # This is the actual eval set.
    test_dataset = IndexingTrainDataset(path_to_data=args.test_data,
                                        max_length=L,
                                        tokenizer=tokenizer,
                                        cache_dir=args.cache_dir,
                                        max_samples = args.test_samples)
    print(train_dataset.total_len, test_dataset.total_len)
    ################################################################
    # docid generation constrain, we only generate integer docids.
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
    # INT_TOKEN_IDS.append(tokenizer.bos_token_id)
    print("Docid tokens:", [tokenizer.decode(i) for i in INT_TOKEN_IDS])

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################

    cnt = 0
    for dp in train_dataset:
      if cnt == 0:
        print(tokenizer.decode(dp[0]), dp[1])
        break

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        learning_rate=args.lr,
        warmup_steps=1000, #10000
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size, #128
        per_device_eval_batch_size=args.batch_size, #128
        eval_strategy ='steps',
        eval_steps=args.eval_steps, #1000
        num_train_epochs=args.num_train_epochs,
        dataloader_drop_last=False,  # necessary
        report_to="none",
        logging_steps=100,
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_Hits@10",
        dataloader_num_workers=6,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # max_steps = args.max_steps
    )
    training_args.id_max_length = args.id_max_length

    trainer = IndexingTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding='longest',
        ),
        compute_metrics=make_compute_metrics(tokenizer, train_dataset.valid_ids),
        callbacks=[QueryEvalCallback(test_dataset, logger, restrict_decode_vocab, training_args, tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab,
        id_max_length=args.id_max_length
    )
    
    trainer.train()
    preds, labels, metrics = trainer.predict(test_dataset)
    predictions = tokenizer.batch_decode(preds[:,:,1])
    with open(os.path.join(args.save_dir, "predictions.txt"), "w") as f:
        f.write("\n".join(predictions))
    
    print(metrics)
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    trainer.save_model()


if __name__ == "__main__":
    main()
