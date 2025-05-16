from io import open
import torch
from torch.utils.data import Dataset
import preprocess
import random
import numpy as np

class MyDataset(Dataset):
    def __init__(self, seperator_chars:list[str], data_locations:list):
        self.data_locations:list = data_locations
        self.labels_unique:list[str] = [] # unique list of labels (use list to preserve order, then dict to remove duplicates)

        self.data_list:list[str] = [] # list of messages
        self.data_list_tensors:list[torch.Tensor] = [] # list of messages in tensor lists
        self.label_list:list[str] = [] # list of labels (ham or spam)
        self.label_list_tensors:list[torch.Tensor] = [] # list of labels (ham or spam) in tensors

        self.spam_index_list:list[int] = []
        self.ham_index_list:list[int] = []

        # read the data from the csv file
        self.labels_unique = ['ham', 'spam']
        for file_index, file in enumerate(data_locations): # go thru each file you input
            lines = open(file, encoding = 'utf-8').read().strip().split('\n')
            for line_index, line in enumerate(lines): # go thru each line and add stuff to lists
                temp_list = line.split(seperator_chars[file_index])
                label = temp_list[0]
                data = ''.join(temp_list[1:])

                self.data_list.append(preprocess.unicode_to_ascii(data))
                self.data_list_tensors.append(preprocess.string_to_tensor(self.data_list[line_index]))
                self.label_list.append(label)
                self.label_list_tensors.append(torch.tensor([self.labels_unique.index(self.label_list[line_index])]))

                # track where the spam and ham is
                if label == 'spam':
                    self.spam_index_list.append(len(self.data_list)-1)
                elif label == 'ham':
                    self.ham_index_list.append(len(self.data_list)-1)
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index:int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[str]]:
        return self.label_list_tensors[index], self.data_list_tensors[index], self.label_list[index], self.data_list[index]
    
def get_batches_from_dataset(subdata:torch.utils.data.Subset, batch_size:int, ham_percent:float) -> list[list[int]]:
    # encountering problem where 85% of data is ham, only 15% is spam so it's missing spam often
        # fix by only using 1/4 of the ham and all the spam AND using a weighted loss function

    # each epoch, get a random 1/4 of the ham and all of the spam, instead of all of the indices in the training data

    subdata_indices = subdata.indices # have to manually recreate what the training dataset's lists WOULD look like since random_split makes Subsets instead of new Datasets
    subdata_spam_index_list = list(set(subdata.dataset.spam_index_list) & set(subdata_indices))
    
    subdata_ham_index_list = list(set(subdata.dataset.ham_index_list) & set(subdata_indices))
    random.shuffle(subdata_ham_index_list)
    subdata_ham_index_list = subdata_ham_index_list[:round(len(subdata_ham_index_list)*ham_percent)]
    
    print(f'{len(subdata_ham_index_list)} ham, {len(subdata_spam_index_list)} spam')
    
    # split training data into batches
    batches = subdata_spam_index_list + subdata_ham_index_list
    random.shuffle(batches)
    batches = np.array_split(batches, round(len(batches) / batch_size)) # split list into batches of indices

    return batches