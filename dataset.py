from io import open
import torch
from torch.utils.data import Dataset
import preprocess

class MyDataset(Dataset):
    def __init__(self, seperator_chars:list[str], data_locations:list):
        self.data_locations:list = data_locations
        self.labels_unique:list[str] = [] # unique list of labels (use list to preserve order, then dict to remove duplicates)

        self.data_list:list[str] = [] # list of messages
        self.data_list_tensors:list[torch.Tensor] = [] # list of messages in tensor lists
        self.label_list:list[str] = [] # list of labels (ham or spam)
        self.label_list_tensors:list[torch.Tensor] = [] # list of labels (ham or spam) in tensors

        # read the data from the csv file
        self.labels_unique = ['ham', 'spam']
        for file_index, file in enumerate(data_locations): # go thru each file you input
            lines = open(file, encoding = 'utf-8').read().strip().split('\n')
            for line_index, line in enumerate(lines): # go thru each line and add stuff to lists
                temp_list = line.split(seperator_chars[file_index])
                label = temp_list[0]
                data = ''.join(temp_list[1:])

                # print(str(file_index) + ' ' + str(line_index) + ' ' + str(temp_list))

                self.data_list.append(preprocess.unicode_to_ascii(data))
                self.data_list_tensors.append(preprocess.string_to_tensor(self.data_list[line_index]))
                self.label_list.append(label)
                self.label_list_tensors.append(torch.tensor([self.labels_unique.index(self.label_list[line_index])]))
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index:int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[str]]:
        return self.label_list_tensors[index], self.data_list_tensors[index], self.label_list[index], self.data_list[index]
            
# all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection'])
# print(len(all_data))