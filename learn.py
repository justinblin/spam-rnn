import torch
from dataset import MyDataset
import preprocess
import postprocess
from rnn import MyRNN
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
rnn = MyRNN(len(preprocess.allowed_char), 512, len(all_data.labels_unique))

# train neural network
def train(rnn:MyRNN, training_data:torch.utils.data.Subset, num_epoch:int = 10, batch_size:int = 64, target_loss:float = 0.05, 
          learning_rate:float = 0.05, criterion = nn.NLLLoss(), show:bool = True):
    # track loss over time
    current_loss = 0
    all_losses = []
    rnn.train() # flag that you're starting to train now
    optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate, momentum = 0.5) # stochastic gradient descent
        # momentum uses previous steps in the current step, faster training by reducing oscillation

    training_data_indices = training_data.indices # have to manually recreate what the training dataset's lists WOULD look like since random_split makes Subsets instead of new Datasets
    training_data_spam_index_list = list(set(training_data.dataset.spam_index_list) & set(training_data_indices))

    print(f'Start training on {len(training_data)} examples')

    # go thru each epoch
    for epoch_index in range(num_epoch):
        # encountering problem where 85% of data is ham, only 15% is spam so it's missing spam often
            # fix by only using 2/3 of the ham and all the spam AND using a weighted loss function

        # each epoch, get a random 2/3 of the ham and all of the spam, instead of all of the indices in the training data
        
        training_data_ham_index_list = list(set(training_data.dataset.ham_index_list) & set(training_data_indices))
        random.shuffle(training_data_ham_index_list)
        training_data_ham_index_list = training_data_ham_index_list[:round(len(training_data_ham_index_list)*1/4)]
        print(f'{len(training_data_ham_index_list)} ham, {len(training_data_spam_index_list)} spam')
        
        # split training data into batches
        batches = training_data_spam_index_list + training_data_ham_index_list
        random.shuffle(batches)
        batches = np.array_split(batches, round(len(batches) / batch_size)) # split list into batches of indices

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
                # print(f'batch index: {batch_index} num batches: {len(batches)}, num batches tenth: {round(len(batches)/10)}')
                print(f'{(int)(batch_index/len(batches)*100)}% complete, loss for current batch: {batch_loss.item() / len(batch)}')

        # log the current loss
        current_loss /= len(batches)
        all_losses.append(current_loss)
        print(f'\nFINISH EPOCH {epoch_index}: average batch loss = {all_losses[-1]}\n')

        # cut early if you reach the goal
        if current_loss < target_loss:
            return all_losses

        current_loss = 0 # reset loss so it doesn't build up in the tracking

    # show training results
    if show:
        plt.figure()
        plt.plot(all_losses)
        plt.show()

    return all_losses

all_losses = train(rnn, train_set, num_epoch = 20, criterion = nn.NLLLoss(weight = torch.tensor([.33, .67])))

torch.save(rnn, "./my_model")

# TEST NEURAL NETWORK
def test(rnn:MyRNN, testing_data:MyDataset, classes:list[str], show:bool = True):
    print(f'Start testing on {len(testing_data)} examples')

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
    percent_correct = (percent_correct*100)//len(testing_data)

    print(torch.round(confusion_matrix, decimals = 2))
    print(str(percent_correct) + "% correct")

    # plot the confusion matrix
    if show:
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

test(rnn, test_set, all_data.labels_unique)