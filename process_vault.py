import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from k_means_constrained import KMeansConstrained

from typing import List, Tuple

from datasets import load_from_disk, load_dataset, concatenate_datasets
import argparse
import os
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--doc_column', type=str, default="code")
parser.add_argument('--query_column', type=str, default="docstring")
parser.add_argument('--index_retrieval_ratio', type=float, default=32)
parser.add_argument('--train_samples', type=int, default=-1)
parser.add_argument('--test_samples', type=int, default=-1)
parser.add_argument('--track_metadata', action="store_true")
parser.add_argument('--semantic_id', action="store_true")
parser.add_argument('--num_cluster', type=int, default=10, help="Only semantic_id=true, Number of clusters at each level")
parser.add_argument('--min_cluster_size', type=int, default=100, help="Only semantic_id=true, Minimum size of each cluster")
args = parser.parse_args()

class CodeBERTSentenceEncoder:
    def __init__(self, model_name="microsoft/unixcoder-base", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True,
                                     max_length=256, return_tensors='pt').to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)
                pooled = self.mean_pooling(output, encoded['attention_mask'])

            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings)

class SemanticIDGenerator:
    def __init__(self, 
                 k: int = args.num_cluster, 
                 c: int = args.min_cluster_size, 
                 random_state: int = 42):
        self.k = k
        self.c = c
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> List[str]:
        """
        Args:
            X: (N, D) numpy array of sentence embeddings

        Returns:
            List of semantic ID strings like "231"
        """
        N = X.shape[0]
        print(f"Number of samples: {N}")
        
        doc_indices = np.arange(N)
        assigned = self._recursive_cluster(X, doc_indices, prefix="")

        doc_ids = [""] * N
        for idx, id_str in assigned:
            doc_ids[idx] = id_str
        return doc_ids

    def _recursive_cluster(self, X: np.ndarray, 
                           indices: np.ndarray, 
                           prefix: str) -> List[Tuple[int, str]]:
        
        if len(indices) <= self.c:
            print(len(indices))
            return [(idx, prefix + str(i)) for i, idx in enumerate(indices)]

        k = min(self.k, len(indices))

        cluster_size = len(indices) // k
        # kmeans = MiniBatchKMeans(n_clusters=k, 
        #                          random_state=self.random_state, 
        #                          batch_size=1024)
        kmeans = KMeansConstrained(n_clusters=k,
                                    size_min=cluster_size,
                                    size_max=cluster_size + 1,
                                    random_state=42)
                               
        labels = kmeans.fit_predict(X[indices])

        results = []
        for cluster_id in range(k):
            sub_indices = indices[labels == cluster_id]
            if len(sub_indices) == 0:
                continue
            sub_prefix = prefix + str(cluster_id)
            results.extend(self._recursive_cluster(X, 
                                                   sub_indices, 
                                                   sub_prefix))
        return results

def main():

    def process_query(query):
        return "Query: " + " ".join(query.lower().split())

    def process_doc(code):
        return "Code: " + code

    if os.path.isdir(args.data_path):
        dataset = load_from_disk(args.data_path)
    else:
        dataset = load_dataset("json", data_files=args.data_path)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    language = args.data_path.split("/")[-1].split(".")[0]

    if args.train_samples == -1:
        trainset = dataset["train_small"]
    else: 
        trainset = dataset["train_small"].select(range(args.train_samples))
    if args.test_samples == -1:
        testset = dataset["test"]
    else:
        testset = dataset["test"].select(range(args.test_samples))
        
    columns = trainset.column_names
    
    keep_metadata = []
    if args.track_metadata:
        keep_metadata = ['hexsha', 'repo', 'path', 'identifier', 'parameters']
        columns = [x for x in columns if x not in keep_metadata]
    
    stop_words = []
    
    print(len(trainset), len(testset))

    trainset = trainset.map(lambda example: {"query": process_query(example[args.query_column]), "doc": process_doc(example[args.doc_column])})
    testset = testset.map(lambda example: {"query": process_query(example[args.query_column]), "doc": process_doc(example[args.doc_column])})

    trainset = trainset.add_column("text_id", ["train_" + str(i) for i in range(len(trainset))]).remove_columns(columns)
    testset = testset.add_column("text_id", ["test_" + str(i) for i in range(len(testset))]).remove_columns(columns)

    merged_dataset = concatenate_datasets([trainset, testset]).shuffle(seed=42)

    print(merged_dataset.column_names)
    print(trainset.column_names)
    print(testset.column_names)
    train_retrieval = []
    train_indexing = []
    test_data = []
    text_based_id_pool = []
    url_based_id_pool = []
    cnt = 0

    ###
    if args.semantic_id:
        text_list = merged_dataset['doc']
        encoder = CodeBERTSentenceEncoder()
        embeddings = encoder.encode(text_list, batch_size=64, normalize=True)
        ssi = SemanticIDGenerator()
        semantic_ids = ssi.fit(embeddings)
    else:
        semantic_ids = [str(i) for i in range(len(merged_dataset))]
    ###

    for dp, sem_id in zip(merged_dataset, semantic_ids):
        dp_r = {x: dp[x] for x in keep_metadata}
        dp_r["text_id"]= str(sem_id)
        text_based_id = "{}({})|{}|{}".format(dp["identifier"], ",".join(x["param"] for x in dp["parameters"]), dp["repo"], dp["path"])
        
        if text_based_id in text_based_id_pool: #remove duplicate function
            continue
        text_based_id_pool.append(text_based_id)
        
        dp_r["url_based_id"] = "/".join([dp["repo"], dp["path"], dp["identifier"] + "(" + ",".join(x["param"] for x in dp["parameters"])+ ")" ])        
        
        assert dp_r["url_based_id"] not in url_based_id_pool, dp_r["url_based_id"]
        url_based_id_pool.append( dp_r["url_based_id"])
        
        
        if dp["text_id"].startswith("train"):
            dp_r["text"]= dp["query"]
            train_retrieval.append(dp_r)
            
            dp_q = dp_r.copy()
            dp_q["text"]= dp["doc"]
            train_indexing.append(dp_q)
        else:            
            dp_r["text"]= dp["query"]
            test_data.append(dp_r)
            
            dp_q = dp_r.copy()
            dp_q["text"]= dp["doc"]
            train_indexing.append(dp_q)
        cnt += 1
    
        
    train_retrieval = train_retrieval[:int(len(train_indexing)/args.index_retrieval_ratio)]

    print(len(train_retrieval), len(train_indexing), len(train_indexing)/len(train_retrieval))
    train_data = train_retrieval + train_indexing

    with jsonlines.open(os.path.join(args.save_dir, f"{language}_train_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(train_data)
        
    with jsonlines.open(os.path.join(args.save_dir, f"{language}_test_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(test_data)

main()