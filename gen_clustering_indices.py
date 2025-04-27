# python script to save indices sorted by solution length
import argparse
import torch
import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
# import pickle
from datasets import load_dataset
# import numpy as np

def extract_question(datum, question_key):
    """Return the raw question string given a dataset example and its config."""
    return datum.get(question_key, "").strip()

def get_embeddings(raw_inputs):
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_name = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.float16, device_map="auto")

    model.eval()

    inputs = tokenizer(raw_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings.squeeze()

def cluster(embeddings, n_clusters=10):
    # with open("embeddings.pkl", "rb") as f:
    #     embeddings = pickle.load(f)

    # print(type(embeddings))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    clusters = kmeans.fit_predict(embeddings)

    # print(clusters)

    return clusters

def gen_indices(num_clusters):
    # for now, jsonl manually downloaded from huggingface, not pushed to repo to save space
    # https://huggingface.co/datasets/GAIR/LIMO/blob/main/limo.jsonl
    # df = pd.read_json(path_or_buf='limo.jsonl', lines=True)

    ds = load_dataset("GAIR/LIMO", split='train')

    raw_inputs = [
        extract_question(datum, "question") + 
        " [SEP] " + 
        extract_question(datum, "solution") 
        for datum in ds
    ]


    embeddings = [get_embeddings(raw_input) for raw_input in raw_inputs]

    print('starting kmeans now')

    clusters = cluster(embeddings, n_clusters=num_clusters)

    return clusters
    
# if __name__ == '__main__':
#     gen_indices(10)