from io import open
import torch
from torch.utils.data import Dataset
import preprocess

class MyDataset(Dataset):
    def __init__(self, data_location):
        self.data_location = data_location
        self.labels_unique = [] # unique list of labels (use list to preserve order, then dict to remove duplicates)

        self.data_list = [] # list of messages
        self.data_list_tensors = [] # list of messages in tensor lists
        self.label_list = [] # list of labels (ham or spam)
        self.label_list_tensors = [] # list of labels (ham or spam) in tensors

        # read the data from the csv file
        self.labels_unique = ['ham', 'spam']
        lines = open(data_location, encoding = 'utf-8').read().strip().split('\n')
        for line_index, line in enumerate(lines): # go thru each line and add stuff to lists
            temp_list = line.split(',')
            label = temp_list[0]
            data = temp_list[1]

            # print(label)

            self.data_list.append(preprocess.unicode_to_ascii(data))
            self.data_list_tensors.append(preprocess.string_to_tensor(self.data_list[line_index]))
            self.label_list.append(label)
            self.label_list_tensors.append(torch.tensor([self.labels_unique.index(self.label_list[line_index])]))
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index:int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[str]]:
        return self.label_list_tensors[index], self.data_list_tensors[index], self.label_list[index], self.data_list[index]
            