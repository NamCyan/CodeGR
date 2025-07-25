from dataclasses import dataclass

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from tqdm import tqdm

class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            max_samples: int=-1,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            cache_dir=cache_dir
        )['train']
        
        if max_samples != -1:
            self.train_data = self.train_data.take(max_samples)

        self.max_length = max_length
        if max_length == -1:
            self.max_length = tokenizer.max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            self.valid_ids.add(str(data['text_id']))


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]

        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['text_id'])


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels