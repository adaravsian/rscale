## Data Selection

The indices are already generated and in the repo, but if you want to replicate the process, here is how.

### Rand Indices
1. Simply select `--method rand` for `train.py`, and the indices selected will be random.

### Length Indices
1. Download the dataset as a .jsonl file at [this link](https://huggingface.co/datasets/GAIR/LIMO/blob/main/limo.jsonl), and make sure it is named `limo.jsonl`. 
2. Run `python gen_len_indices.py`, and the indices will be generated to the filename printed out (should be `indices/len_indices.txt`).

### Clustering Indices
1. Run `python gen_clustering_indices.py --proportion p`, where p is some float proportion in the range (0, 1). This corresponds to how many cluster centers will be used in the K Means clustering.
2. Repeat for each proportion you intend to try out on.
3. The indices will be generated to `indices/cluster_indices_{proportion}.txt`, where proportion is the proportion you selected.