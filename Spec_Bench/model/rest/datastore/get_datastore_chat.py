from datasets import load_dataset
from transformers import AutoTokenizer
import draftretriever
from tqdm import tqdm
import json

import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-path",
    type=str,
    default="lmsys/vicuna-7b-v1.5",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--large-datastore",
    type=bool,
    default=False,
    help="Whether to use a large datastore",
)
args = parser.parse_args()
print(args)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)


datastore_path = './datastore_chat_large.idx' if args.large_datastore else './datastore_chat_small.idx'
writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512*1024*1024,
    vocab_size=tokenizer.vocab_size,
)
if args.large_datastore:
    dataset = load_dataset('stingning/ultrachat', split='train')
    total_length = len(dataset)
    print("number of samples: ", total_length)
    for conversations in tqdm(dataset, total=total_length):
        for sample in conversations['data']:
            token_list = tokenizer.encode(sample)
            writer.add_entry(token_list)
else:

    dataset_path = 'datastore/dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json'
    assert dataset_path is not None, "please download the dataset from https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered"
    dataset = json.load(open(dataset_path))
    total_length = len(dataset)
    print("number of samples: ", total_length)
    for conversations in tqdm(dataset, total=total_length):
        for sample in conversations['conversations']:
            token_list = tokenizer.encode(sample['value'])
            writer.add_entry(token_list)

writer.finalize()

