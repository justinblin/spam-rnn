import torch
import torch.nn as nn

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import MyDataset, get_batches_from_dataset


def find_loss(model, criterion, data_subset:torch.utils.data.Subset, batches:list[list[int]]) -> float:
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

def find_best_lr(model, criterion, validating_data:torch.utils.data.Subset, ham_percent:float, #batches:list[list[int]], 
                 batch_size:int = 64, num_batches:int = 8, low_bound = 0.001, num_steps = 10, step_size = 2, show = True) -> float:
    if show: print('\nSTART FINDING BEST LR\n')
    
    torch.save(model, './my_model')
    
    loss_dict:dict[float:float] = {} # map lr to loss (could just have a single counter for max improvement instead of tracking all)
    batches = get_batches_from_dataset(validating_data, batch_size, ham_percent)
    if len(batches) > num_batches:
        batches = batches[:num_batches]
    if show: print(f'use {len(batches)} batches of {batch_size} elements')

    curr_lr = low_bound
    for index in range(num_steps): # go through the  the lr's exponentially
        print('progress: ', end='') # start progress bar

        # FIND THE LOSS BEFORE
        current_loss = find_loss(model, criterion, validating_data, batches)


        # DO BACK PROPOGATION AND FIND THE LOSS AFTER
        optimizer = torch.optim.SGD(model.parameters(), lr = curr_lr, momentum = 0.5, weight_decay=0.01)

        new_loss = 0 # average loss for all the batches

        for batch_index, batch in enumerate(batches): # go thru each batch
            batch_loss = 0 # total loss for this batch
            # go thru each tensor in this batch
            for curr_elem in batch:
                # run forward and figure out the loss
                (label_tensor, name_tensor, label, name) = validating_data.dataset[curr_elem]
                output = model(name_tensor) # tensor that's outputted
                loss = criterion(output, label_tensor)
                batch_loss += loss

            batch_loss.backward() # find out how much to change each weight/bias
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step() # apply the changes to weight/biases
            optimizer.zero_grad() # prevent exploding gradients

            new_loss += batch_loss.item() / len(batch) # add average loss for this batch into current_loss

            if batch_index % max(round(len(batches)/10),1) == 0: # progress bar (prevent /0 error)
                print('[]', end='')

        print() # end progress bar

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
    pass
    # all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection']) # 11147 total testcases
    # train_set, validating_set, test_set = torch.utils.data.random_split(all_data, [.6, .2, .2])
    
    # class_weights:list[float] = [0.33, 0.67]

    # rnn = torch.load('./my_model', weights_only = False)
    # criterion = nn.NLLLoss(weight = torch.tensor(class_weights))

    # batches = get_batches_from_dataset(validating_set, 64, 0.25)

    # print(find_best_lr(rnn, criterion, validating_set, 0.25))

if __name__ == "__main__":
    main()