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
                 prev_lr:float, batch_size:int = 64, num_batches:int = 8, 
                 low_bound = 0.001, num_steps = 10, step_size = 2, show = True) -> float:
    if show: print('\nSTART FINDING BEST LR\n')
    
    torch.save(model, './my_model')
    
    best_lr:float = 0
    best_loss_improvement:float = -100

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
        optimizer = torch.optim.SGD(model.parameters(), lr = curr_lr, momentum = 0.5, weight_decay=0.05)

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
        if current_loss - new_loss > best_loss_improvement: # if the cur improvement is better than prev best, make it the best
            best_loss_improvement = current_loss - new_loss
            best_lr = curr_lr

        # restore the model
        model = torch.load('./my_model', weights_only = False)

        if show: print(f'learning rate: {curr_lr}, loss improvement: {current_loss - new_loss}, old loss: {current_loss}, new loss: {new_loss}')

        curr_lr *= step_size

    final_lr = (best_lr + 2*prev_lr)/3
    print(f'Best LR before momentum: {best_lr}, after momentum: {final_lr}\n')
    return final_lr # sorta like a momentum for lr so it doesn't change as violently

def main():
    all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection']) # 11147 total testcases
    train_set, validation_set, test_set = torch.utils.data.random_split(all_data, [.6, .2, .2])
    

    rnn = torch.load('./my_model', weights_only = False)
    criterion = nn.NLLLoss(weight = torch.tensor([1., 5.]))

    print(find_best_lr(rnn, criterion, validation_set, 0.15))

if __name__ == "__main__":
    main()