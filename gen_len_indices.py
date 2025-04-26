# python script to save indices sorted by solution length
import pandas as pd

if __name__ == '__main__':
    # jsonl manually downloaded from huggingface, not pushed to repo to save space
    df = pd.read_json(path_or_buf='limo.jsonl', lines=True)
    # minidf = df[:10]
    minidf = df
    # print(df.iloc[0])
    # print(minidf)
    len_indices = minidf['solution'].str.len().sort_values(ascending=False).index
    # print(minidf['solution'].str.len())
    # print(type(len_indices))
    data = len_indices._data
    with open('indices/len_indices.txt', 'w') as f:
        for num in data:
            f.write(str(num) + ', ')
    sorted_minidf = minidf.reindex(len_indices)
    # print(sorted_minidf['solution'].str.len())
    indices = None
    with open('indices/len_indices.txt', 'r') as f:
        indices = f.read()
    # print(indices)