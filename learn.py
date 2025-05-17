import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import MyDataset, get_batches_from_dataset
import preprocess
import postprocess
from rnn import MyRNN
from learning_rate_finder import find_best_lr, find_loss

# train neural network
def train(rnn:MyRNN, training_data:torch.utils.data.Subset, testing_data:torch.utils.data.Subset, ham_percent:float, 
          num_epoch:int = 10, batch_size:int = 64, target_loss:float = 0.05, learning_rate:float = 0.064, 
          criterion = nn.NLLLoss(), show_graph:bool = True, dynamic_lr:bool = True) -> tuple[list[float]]:
    # track loss over time
    train_losses = []
    test_losses = []
    learning_rates = []
    rnn.train() # flag that you're starting to train now

    print(f'\nStart training on {len(training_data)} examples\n')

    # go thru each epoch
    for epoch_index in range(num_epoch):
        print(f'start epoch {epoch_index}, learning rate: {learning_rate}')

        optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate, momentum = 0.5) # stochastic gradient descent
            # momentum uses previous steps in the current step, faster training by reducing oscillation

        batches = get_batches_from_dataset(training_data, batch_size, ham_percent)

        current_loss = 0 # reset loss so it doesn't build up in the tracking

        # go thru each batch
        for batch_index, batch in enumerate(batches):
            batch_loss = 0 # total loss for this batch
            # go thru each tensor in this batch
            for curr_elem in batch:
                # run forward and figure out the loss
                (label_tensor, name_tensor, label, name) = training_data.dataset[curr_elem]
                output = rnn(name_tensor) # tensor that's outputted
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # run back propogation
            batch_loss.backward() # find out how much to change each weight/bias
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step() # apply the changes to weight/biases
            optimizer.zero_grad() # prevent exploding gradients

            current_loss += batch_loss.item() / len(batch) # add average loss for this batch into current_loss

            # show progress (10 per epoch)
            if batch_index % round(len(batches)/10) == 0:
                print(f'{(int)(batch_index/len(batches)*100)}% complete, loss for current batch: {batch_loss.item() / len(batch)}')

        # log the current learning rate
        learning_rates.append(learning_rate)

        # check testing loss and add to list
        test_losses.append(find_loss(rnn, criterion, testing_data, batches))

        # log the current loss
        current_loss /= len(batches)
        train_losses.append(current_loss)
        print(f'\nFINISH EPOCH {epoch_index}: training average batch loss = {train_losses[-1]}, testing loss = {test_losses[-1]}\n')

        # cut early if you reach the goal
        if test_losses[-1] < target_loss:
            break

        # look for a new lr if there's a loss plateau
        # check the loss every 3 epochs (exclude idx 0), if it isn't >=10% better than the last time, find a new lr
        if epoch_index % 3 == 0:
            torch.save(rnn, './my_model') # save model every 3 epochs
            if dynamic_lr and epoch_index != 0 and train_losses[epoch_index] > train_losses[epoch_index-3]*0.9:
                learning_rate = find_best_lr(rnn, criterion, training_data, ham_percent)

    # show training results
    if show_graph:
        plt.figure()
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.plot(learning_rates)
        plt.legend(['train loss', 'test loss', 'learning rates'])
        plt.show()

    print(f'train_losses = {train_losses}')
    print(f'test_losses = {test_losses}')
    print(f'learning_rates = {learning_rates}')
    torch.save(rnn, "./my_model")
    return train_losses, test_losses, learning_rates

# TEST NEURAL NETWORK
def test(rnn:MyRNN, testing_data:MyDataset, classes:list[str], show_graph:bool = True):
    print(f'\nStart testing on {len(testing_data)} examples\n')

    confusion_matrix = torch.zeros(len(classes), len(classes))
    percent_correct = 0

    rnn.eval() # turn on eval flag
    with torch.no_grad(): # don't record gradients
        for index in range(len(testing_data)): # go thru each test example
            (label_tensor, data_tensor, label, data) = testing_data[index]
            label_index = classes.index(label) # the index of the correct answer for this testcase

            # print(data)
            # print(all_data.data_list.index(data))
            output = rnn(data_tensor)
            guess, guess_index = postprocess.label_from_output(output, classes) # the guess and index of the guess

            if index % (round(len(testing_data)/10)) == 0: # print 10 outputs
                print(f"testcase index: {index}, guess: {str(guess)}, guess_idx: {str(guess_index)}, label: {str(label)}, label_idx: {str(label_index)}")

            confusion_matrix[guess_index][label_index] += 1
            if guess_index == label_index:
                percent_correct += 1

    # turn the total count into percentage (aka normalize)
    graph_total = confusion_matrix.sum()
    if graph_total > 0:
        confusion_matrix /= graph_total
    percent_correct = (percent_correct*100)/len(testing_data)
    precision = confusion_matrix[1][1]/sum(confusion_matrix[1]) # AKA when you guess spam, how many were right
    recall = confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]) # AKA of all the spam, how many did you guess right

    print(torch.round(confusion_matrix, decimals = 3))
    print(f'{percent_correct}% correct')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1 score: {2*precision*recall/(precision+recall)}')

    # plot the confusion matrix
    if show_graph:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion_matrix.cpu().numpy()) # need to convert from gpu to cpu mem bcs numpy uses cpu
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
        ax.set_yticks(np.arange(len(classes)), labels=classes)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.xlabel("Answer")
        plt.ylabel("Guess")
        plt.show()

def main():
    # use GPU if possible
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    torch.set_default_device(device)

    print(device)

    # SETUP DATASET
    all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection']) # 11147 total testcases
    train_set, test_set, extra_set = torch.utils.data.random_split(all_data, [.8, .2, .0])

    # CREATE/TRAIN NN
    from_scratch:bool = False # train a new model OR keep training a previous model
    train_model:bool = False
    test_model:bool = True

    if from_scratch: rnn = MyRNN(len(preprocess.allowed_char), 512, len(all_data.labels_unique))
    else: rnn = torch.load('./my_model', weights_only = False) # train off a pretrained model instead of from scratch
    criterion = nn.NLLLoss(weight = torch.tensor([.33, .67]))
    ham_percent = 0.25

    if train_model:
        if from_scratch: best_lr = 0.064
        else: best_lr = find_best_lr(rnn, criterion, train_set, ham_percent)
        train_losses, test_losses, learning_rates = train(rnn, train_set, test_set, ham_percent, num_epoch = 100, learning_rate = best_lr, criterion = criterion)

    if test_model: test(rnn, test_set, all_data.labels_unique)

if __name__ == "__main__":
    main()