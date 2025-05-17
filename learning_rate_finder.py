import torch
import torch.nn as nn

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import MyDataset, get_batches_from_dataset
from rnn import MyRNN
import preprocess


def find_loss(model:MyRNN, criterion, data_subset:torch.utils.data.Subset, batches:list[list[int]]) -> float:
    current_loss = 0 # average loss for all the batches

    for batch_index, batch in enumerate(batches): # go thru each batch
        batch_loss = 0 # total loss for this batch
        # go thru each tensor in this batch
        for curr_elem in batch:
            # run forward and figure out the loss
            (label_tensor, name_tensor, label, name) = data_subset.dataset[curr_elem]
            output = model(name_tensor) # tensor that's outputted
            loss = criterion(output, label_tensor)
            batch_loss += loss

        current_loss += batch_loss.item() / len(batch) # add average loss for this batch into current_loss

    current_loss /= len(batches)

    return current_loss

def find_best_lr(model:MyRNN, criterion, training_data:torch.utils.data.Subset, ham_percent:float, 
                 batch_size:int = 64, num_batches:int = 8, low_bound = 0.001, num_steps = 11, step_size = 2, show = True) -> float:
    if show: print('\nSTART FINDING BEST LR\n')
    
    torch.save(model, './my_model')
    
    loss_dict:dict[float:float] = {} # map lr to loss
    batches = get_batches_from_dataset(training_data, batch_size, ham_percent)
    if len(batches) > num_batches:
        batches = batches[:num_batches]
    if show: print(f'use {len(batches)} batches of {batch_size} elements')

    curr_lr = low_bound
    for index in range(num_steps): # go through the  the lr's exponentially
        # FIND THE LOSS BEFORE
        current_loss = find_loss(model, criterion, training_data, batches)


        # DO BACK PROPOGATION AND FIND THE LOSS AFTER
        optimizer = torch.optim.SGD(model.parameters(), lr = curr_lr, momentum = 0.5)

        new_loss = 0 # average loss for all the batches

        for batch_index, batch in enumerate(batches): # go thru each batch
            batch_loss = 0 # total loss for this batch
            # go thru each tensor in this batch
            for curr_elem in batch:
                # run forward and figure out the loss
                (label_tensor, name_tensor, label, name) = training_data.dataset[curr_elem]
                output = model(name_tensor) # tensor that's outputted
                loss = criterion(output, label_tensor)
                batch_loss += loss

            batch_loss.backward() # find out how much to change each weight/bias
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step() # apply the changes to weight/biases
            optimizer.zero_grad() # prevent exploding gradients

            new_loss += batch_loss.item() / len(batch) # add average loss for this batch into current_loss

        new_loss /= len(batches)


        # RECORD THE DIFFERENCE AND RESTORE THE MODEL FOR THE NEXT LR
        loss_dict[curr_lr] = current_loss - new_loss # stick loss difference in the loss dict

        # restore the model
        model = torch.load('./my_model', weights_only = False)

        if show: print(f'learning rate: {curr_lr}, loss improvement: {current_loss - new_loss}, old loss: {current_loss}, new loss: {new_loss}')

        curr_lr *= step_size

    # return the key that had the largest loss difference (pick the largest one if ties)
    loss_dict_list = []
    for key, value in loss_dict.items(): # turn dict into list
        loss_dict_list.append((key, value))
    for index in range(len(loss_dict_list)-1, -1, -1): # go backwards thru list
        if loss_dict_list[index][1] == max(loss_dict.values()):
            if show: print(f'\nbest learning rate: {loss_dict_list[index][0]}\n')
            return loss_dict_list[index][0]
    return None

def main():
    all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection']) # 11147 total testcases
    train_set, test_set, extra_set = torch.utils.data.random_split(all_data, [.8, .2, .0])
    
    class_weights:list[float] = [0.33, 0.67]

    rnn = MyRNN(len(preprocess.allowed_char), 512, len(all_data.labels_unique))
    criterion = nn.NLLLoss(weight = torch.tensor(class_weights))

    print(find_best_lr(rnn, criterion, train_set, 0.25))

if __name__ == "__main__":
    main()