from io import open
import torch
from torch.utils.data import Dataset
import preprocess
import random
import numpy as np

class MyDataset(Dataset):
    def __init__(self, seperator_chars:list[str], data_locations:list):
        print(f'\nstart preparing dataset')

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
        
        print(f'dataset has {len(self.spam_index_list)} spam and {len(self.ham_index_list)} ham\n')
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index:int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[str]]:
        return self.label_list_tensors[index], self.data_list_tensors[index], self.label_list[index], self.data_list[index]
    
def get_batches_from_dataset(subdata:torch.utils.data.Subset, batch_size:int, ham_percent:float = 1.0, spam_extra:float = 0.0, num_batches:int = None) -> list[list[int]]:
    """
    If num_batches is used, batch_size will be ignored
    """
    
    # encountering problem where 85% of data is ham, only 15% is spam so it's missing spam often
        # fix by only using 1/4 of the ham and all the spam AND using a weighted loss function

    # each epoch, get a random 1/4 of the ham and all of the spam, instead of all of the indices in the training data

    subdata_indices = subdata.indices # have to manually recreate what the training dataset's lists WOULD look like since random_split makes Subsets instead of new Datasets
    subdata_spam_index_list = list(set(subdata.dataset.spam_index_list) & set(subdata_indices))
    print(f'pre-ROS/RUS: {len(subdata_spam_index_list)} spam', end=', ')
    random.shuffle(subdata_spam_index_list)
    subdata_spam_index_list += subdata_spam_index_list[:round(len(subdata_spam_index_list)*spam_extra)]
    
    subdata_ham_index_list = list(set(subdata.dataset.ham_index_list) & set(subdata_indices))
    print(f'{len(subdata_ham_index_list)} ham')
    random.shuffle(subdata_ham_index_list)
    subdata_ham_index_list = subdata_ham_index_list[:round(len(subdata_ham_index_list)*ham_percent)]
    
    print(f'post-ROS/RUS: {len(subdata_spam_index_list)} spam, {len(subdata_ham_index_list)} ham')
    
    # split training data into batches
    batches = subdata_spam_index_list + subdata_ham_index_list
    random.shuffle(batches)
    batches = np.array_split(batches, num_batches if num_batches else round(len(batches) / batch_size)) # split list into batches of indices

    return batches

def main():
    print(torch.__version__)

    # use GPU if possible
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    torch.set_default_device(device)

    print(device)

    all_data = MyDataset([',', '\t', ','], 
                         ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection', # first 2 was 11147 total testcases
                          'data/Mendeley Data collection/Dataset_5971.csv']) # total roughly 17000 testcases
    train_set, validation_set, test_set = torch.utils.data.random_split(all_data, [1, 0, 0], generator=torch.Generator(device=device))

    get_batches_from_dataset(train_set, len(train_set), 0.75, 1.0, 1)

if __name__ == "__main__":
    main()