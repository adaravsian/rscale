## Running

### Training
To train a model yourself, run  
`rm -rf models/"$NAME"`  
`accelerate launch
  --multi_gpu
  --num_processes $NUM_PROCS
  train.py
    --proportion $PROPORTION
    --method $METHOD
    --name $NAME`  
and replace variables as appropriately.

`$PROPORTION` is a float between 0 and 1, representing how much of the dataset is being selected.  
`$METHOD` is `rand`, `len`, or `cluster`. This represents how the proportion of the dataset is selected.  
`$NAME` is used for filepaths.


The default model used is [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), which is a gated model. This means you will have to get access to the model by requesting it on Huggingface, and then you have to authenticate your Huggingface account before running. Here are some links for tutorials on [authenticating](https://huggingface.co/docs/hub/en/datasets-polars-auth), [authenticating using the CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), and [using tokens](https://huggingface.co/docs/hub/en/security-tokens).

Alternatively, you can use a different model by changing `BASE_MODEL` in `utils.py`.

### Inference
To run inference, run `python inference.py --name $NAME`.

### Evaluation
To run evaluation, run `python eval.py --name $NAME`.

## Dataset

The dataset used is at https://huggingface.co/datasets/GAIR/LIMO. 

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