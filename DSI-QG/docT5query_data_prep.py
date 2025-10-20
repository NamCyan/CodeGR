from datasets import load_from_disk, load_dataset, concatenate_datasets
import argparse
import os
import jsonlines
# from utils import get_node_by_kind, lang_parser

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--doc_column', type=str, default="code")
parser.add_argument('--query_column', type=str, default="docstring")
parser.add_argument('--index_retrieval_ratio', type=float, default=32)
parser.add_argument('--summarization', action="store_true")
parser.add_argument('--train_samples', type=int, default=-1)
parser.add_argument('--test_samples', type=int, default=5000)
parser.add_argument('--track_metadata', action="store_true")
args = parser.parse_args()

if os.path.isdir(args.data_path):
    dataset = load_from_disk(args.data_path)
else:
    dataset = load_dataset("json", data_files=args.data_path)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

language = args.data_path.split("/")[-1].split(".")[0]
trainset = dataset["train_small"]
testset = dataset["test"]

if args.test_samples != -1:
    testset = testset.shuffle(seed=42)
    testset = testset.take(args.test_samples)
columns = trainset.column_names

keep_metadata = []
if args.track_metadata:
    keep_metadata = ['hexsha', 'repo', 'path', 'identifier', 'parameters']
    columns = [x for x in columns if x not in keep_metadata]

stop_words = []


print(len(trainset), len(testset))

if args.summarization:
    trainset = trainset.map(lambda example: {"text_id": example[args.query_column], "text": f"Generate docstring for this {language} code: {example[args.doc_column]}"}).remove_columns(columns)
    testset = testset.map(lambda example: {"text_id": example[args.query_column], "text": f"Generate docstring for this {language} code: {example[args.doc_column]}"}).remove_columns(columns)
    if args.train_samples != -1:
        trainset = trainset.shuffle(seed=42)
        trainset = trainset.take(args.train_samples)
    if args.test_samples != -1:
        testset = testset.shuffle(seed=42)
        testset = testset.take(args.test_samples)
    print(len(trainset), len(testset))
    trainset.to_json(os.path.join(args.save_dir, f"train/vault_{language}.json"))
    testset.to_json(os.path.join(args.save_dir, f"test/vault_{language}.json"))
else:
    def process_query(query):
        return "Query: " + " ".join(query.lower().split())

    def process_doc(code):
        return "Code: " + code

    trainset = trainset.map(lambda example: {"query": process_query(example[args.query_column]), "doc": process_doc(example[args.doc_column])})
    testset = testset.map(lambda example: {"query": process_query(example[args.query_column]), "doc": process_doc(example[args.doc_column])})

    trainset = trainset.add_column("text_id", ["train_" + str(i) for i in range(len(trainset))]).remove_columns(columns)
    testset = testset.add_column("text_id", ["test_" + str(i) for i in range(len(testset))]).remove_columns(columns)

    merged_dataset = concatenate_datasets([trainset, testset]).shuffle(seed=42)


    train_data = []
    test_data = []
    text_based_id_pool = []
    url_based_id_pool = []
    identifier_based_id_pool= []
    identifier_based_id_pool_dict= {}
    cnt = 0
    for dp in merged_dataset:
        dp_doc = {x: dp[x] for x in keep_metadata}
        dp_doc["text_id"]= str(cnt)
        text_based_id = "{}({})|{}|{}".format(dp["identifier"], ",".join(x["param"] for x in dp["parameters"]), dp["repo"], dp["path"])
        
        if text_based_id in text_based_id_pool: #remove duplicate function
            continue
        text_based_id_pool.append(text_based_id)
        
        dp_doc["url_based_id"] = "/".join([dp["repo"], dp["path"], dp["identifier"] + "(" + ",".join(x["param"] for x in dp["parameters"])+ ")" ])
        # dp_doc["identifier_based_id"] = " ".join([dp["repo"].split("/")[1], dp["path"].split("/")[-1].split(".")[0]] + get_identifier(dp["doc"][5:].strip()))
        
        
        assert dp_doc["url_based_id"] not in url_based_id_pool, dp_doc["url_based_id"]
        url_based_id_pool.append( dp_doc["url_based_id"])
        
        if dp["text_id"].startswith("train"):
            dp_doc["text_id"] = dp["query"][6:].strip()
            dp_doc["text"]= "Generate docstring for code: {}".format(dp["doc"][5:].strip())
            train_data.append(dp_doc)
        else:            
            dp_doc["text_id"] = dp["query"][6:].strip()
            dp_doc["text"]= "Generate docstring for code: {}".format(dp["doc"][5:].strip())
            test_data.append(dp_doc)
        cnt += 1
        

    with jsonlines.open(os.path.join(args.save_dir, f"{language}_train_docT5.json"), mode='w') as writer:
        writer.write_all(train_data)
        
    with jsonlines.open(os.path.join(args.save_dir, f"{language}_test_docT5.json"), mode='w') as writer:
        writer.write_all(test_data)
