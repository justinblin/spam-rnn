import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import MyDataset

def find_best_lr(model, criterion, optimizer, training_data:torch.utils.data.Subset) -> float:
    

    return None

def main():
    # find some way to pass the train_set over here, maybe put it in a file?

    all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection']) # 11147 total testcases
    train_set, test_set, extra_set = torch.utils.data.random_split(all_data, [.8, .2, .0])
    
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