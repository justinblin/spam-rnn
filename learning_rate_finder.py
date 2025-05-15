from torch_lr_finder import LRFinder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
# from learn import train_set
import math
import random
import numpy as np

def find_best_lr(model, criterion, optimizer, trainloader) -> float:
    lr_finder = LRFinder(model, optimizer, criterion, device = 'cpu')
    lr_finder.range_test(trainloader, end_lr = 100, num_iter = 100)
    lr_finder.plot()
    lr_finder.reset()

    print(lr_finder.history)

    return None

class HamSpamBatchSampler(Sampler[list[int]]):
    def __init__(self, data: torch.utils.data.Subset, batch_size: int, percent_ham = 1.) -> None:
        self.data = data
        self.batch_size = batch_size
        self.percent_ham = percent_ham

    # return the number of batches (round up AKA count incomplete batches)
    def __len__(self) -> int:
        return math.ceil(len(self.data)/self.batch_size)
    
    # return list of batches of indices
    def __iter__(self) -> list[list[int]]:
        data_indices = self.data.indices # have to manually recreate what the dataset's lists WOULD look like since random_split makes Subsets instead of new Datasets
        data_spam_index_list = list(set(self.data.dataset.spam_index_list) & set(data_indices))

        # only keep a portion of ham data (based on percent ham)
        data_ham_index_list = list(set(self.data.dataset.ham_index_list) & set(data_indices))
        random.shuffle(data_ham_index_list)
        data_ham_index_list = data_ham_index_list[:round(len(data_ham_index_list)*self.percent_ham)]

        # mix them together and return the final batches
        batches = data_spam_index_list + data_ham_index_list
        random.shuffle(batches)
        batches = np.array_split(batches, round(len(batches) / self.batch_size)) # split list into batches of indices

        print(f'{len(data_ham_index_list)} ham, {len(data_spam_index_list)} spam')

        return batches

def main():
    pass

    # find some way to pass the train_set over here, maybe put it in a file?

    
    # class_weights:list[float] = [0.33, 0.67]
    # learning_rate = 0.05

    # rnn = torch.load('./my_model', weights_only = False)
    # criterion = nn.NLLLoss(weight = torch.tensor(class_weights))
    # optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate, momentum = 0.5)
    # sampler = HamSpamBatchSampler(train_set, 64, 0.25)
    # trainloader = DataLoader(train_set, batch_sampler = sampler)

    # best_lr = find_best_lr(rnn, criterion, optimizer, trainloader)
    # print(best_lr)

if __name__ == "__main__":
    main()