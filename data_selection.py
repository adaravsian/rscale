import pandas as pd
from datasets import load_dataset

"""
Class to select datapoints from a dataset object

dataset is assumed to be a huggingface/arrows dataset object

LIMO dataset bibtex citation
@misc{ye2025limoreasoning,
      title={LIMO: Less is More for Reasoning}, 
      author={Yixin Ye and Zhen Huang and Yang Xiao and Ethan Chern and Shijie Xia and Pengfei Liu},
      year={2025},
      eprint={2502.03387},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03387}, 
}

"""
class DatasetSelector:
    
    """
    @param if no dataset is passed in, the LIMO dataset is used
    """
    def __init__(self, dataset=None):
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = load_dataset("GAIR/LIMO", split="train")
        
        # split train test

        # if type(self.dataset) is datasets.DatasetDict:
        #     try:
        #         _ = self.dataset['test']
        #         ds = self.dataset
        #     except:
        #         ds = self.dataset['train'].train_test_split(test_size=0.15, shuffle=True)
        # else:
        #     ds = self.dataset.train_test_split(test_size=0.15, shuffle=True)

        # self.dataset = ds['train']
        # self.test = ds['test']
        

    """
    select a portion of self.dataset, unless test is specified

    @param the method used to select. 'rand' if random, etc. (will add how to actually select once i make the functions for it)
    @param the proportion of the dataset to select from. if proportion = 1.0, the whole dataset is selected
    """
    def select(self, method='rand', proportion=1.0):
        if method == 'rand':
            return self.__select_random(proportion=proportion)
        
        elif method == 'best':
            # TODO
            # maybe try uncertainty sampling?
            # would need a finished model and its predictions to do this

            # uncertainty w/ entropy
            # given probabilities of predictions from a model:
            # entropies = scipy.stats.entropy(probabilities)
            # and then pick the k samples that correspond with the highest k entropies
            # idk how that would work with whatever model we choose though

            # uncertainty
            # given probabilities of predictions from a model
            # uncertainties = 1 - probabilities.max(axis=1) # might be a different axis depending on shape of probabilities
            # then pick the k samples that correspond with the highest k entropies
            return None
        else:
            print('*' * 40, '\ninvalid method of selection, defaulting to random\n' + '*' * 40)
            return self.__select_random(proportion=proportion)


    """
    @param the proportion of the dataset to select from. if proportion=1.0, the whole dataset is selected
    the selected portion is saved in self.selected_dataset and returned by this method
    """
    def __select_random(self, proportion=1.0):
        dataset = self.dataset

        dataset = dataset.shuffle(seed=42)
        selected = dataset.select(range(int(dataset.num_rows * proportion)))
        self.selected_dataset = selected
        return selected
    

    


        

    



