# python script to save indices sorted by solution length
import pandas as pd

if __name__ == '__main__':
    # jsonl manually downloaded from huggingface, not pushed to repo to save space
    df = pd.read_json(path_or_buf='limo.jsonl', lines=True)

    # minidf = df[:10]
    minidf = df
    
    len_indices = minidf['solution'].str.len().sort_values(ascending=False).index
    
    data = len_indices._data
    filepath = 'indices/len_indices.txt'

    with open(filepath, 'w') as f:
        for num in data:
            f.write(str(num) + ', ')

    print('indices written to', filepath)
    # sorted_minidf = minidf.reindex(len_indices)


    # indices = None
    # with open(filepath, 'r') as f:
    #     indices = f.read()

    
    # print(indices)