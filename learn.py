import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve
import math

from dataset import MyDataset, get_batches_from_dataset
import preprocess
import postprocess
from rnn import MyRNN, MyRNN_4x_Linear_LeakyReLU, MyRNN_Mini_Boi
from learning_rate_finder import find_best_lr, find_loss

# train neural network
def train(rnn, training_data:torch.utils.data.Subset, validating_data:torch.utils.data.Subset, testing_data:torch.utils.data.Subset, 
          ham_percent:float, num_epoch:int = 10, batch_size:int = 64, target_loss:float = 0.08, learning_rate:float = 0.064, 
          criterion = nn.NLLLoss(), show_graph:bool = True, epoch_per_dynamic_lr:int = 3, target_progress_per_epoch:float=0.03, 
          num_batches:int = 8, low_bound:float = 0.001, num_steps:int = 9, print_outlier_batches=False) -> tuple[list[float]]:
    # track loss over time
    train_metrics:tuple[list[float]] = ([],[],[],[],[]) # tuple of lists for loss, accuracy, precision, recall, and f1
    test_metrics:tuple[list[float]] = ([],[],[],[],[])
    learning_rates = []
    abnormal_batches:list[list[int]] = []

    print(f'\nStart training on {len(training_data)} examples\n')

    # go thru each epoch
    for epoch_index in range(num_epoch):
        rnn.train() # flag that you're starting to train now (redo it each epoch bcs testing makes it go away)

        optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate, momentum = 0.5, weight_decay=0.1) # stochastic gradient descent
            # momentum uses previous steps in the current step, faster training by reducing oscillation

        # look for a new lr if there's a loss plateau
        # check the loss every 3 epochs (default), if it isn't >=10% (default) better than the last time, find a new lr
        if epoch_per_dynamic_lr != 0 and epoch_index % epoch_per_dynamic_lr == 0:
            # exceptions for first and second activations since plateau detection needs at least 2 points
            if epoch_index == 0 or epoch_index == epoch_per_dynamic_lr or \
            train_metrics[0][-1] > train_metrics[0][-1-epoch_per_dynamic_lr]*(1 - target_progress_per_epoch*epoch_per_dynamic_lr):
                learning_rate = find_best_lr(rnn, criterion, validating_data, ham_percent, batch_size=batch_size, 
                                             optimizer_param=optimizer, num_batches=num_batches, low_bound=low_bound, num_steps=num_steps)
                
        print(f'start epoch {epoch_index}, learning rate: {learning_rate}')

        current_loss = 0 # reset loss so it doesn't build up in the tracking
        confusion_matrix = torch.zeros(len(training_data.dataset.labels_unique), len(training_data.dataset.labels_unique))

        batches = get_batches_from_dataset(training_data, batch_size, ham_percent) # DO NOT use the same batches for dynamic lr as training

        abnormal_batches.append([])

        # go thru each batch
        for batch_index, batch in enumerate(batches):
            num_spam = 0 # troubleshooting for outlier batches with huge loss

            batch_loss = 0 # total loss for this batch
            # go thru each tensor in this batch
            for curr_elem in batch:
                # run forward and figure out the loss
                (label_tensor, name_tensor, label, name) = training_data.dataset[curr_elem]
                output = rnn(name_tensor) # tensor that's outputted
                loss = criterion(output, label_tensor)
                batch_loss += loss

                if label == 'spam': # troubleshooting for outlier batches with huge loss
                    num_spam += 1

                # get metrics on the training data
                label_index = training_data.dataset.labels_unique.index(label)
                guess, guess_index = postprocess.label_from_output(output, training_data.dataset.labels_unique)
                confusion_matrix[guess_index][label_index] += 1

            # run back propogation
            batch_loss.backward() # find out how much to change each weight/bias
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step() # apply the changes to weight/biases
            optimizer.zero_grad() # prevent exploding gradients

            current_loss += batch_loss.item() / len(batch) # add average loss for this batch into current_loss

            # show progress (10 per epoch)
            if batch_index % max(round(len(batches)/10),1) == 0:
                print(f'{(int)(batch_index/len(batches)*100)}% complete, loss for current batch: {batch_loss.item() / len(batch)}, {num_spam}/{len(batch)} spam')

            # troubleshooting for outlier batches with huge loss (ONLY FOR FINE TUNING)
            if print_outlier_batches and batch_loss.item() / len(batch) > 0.05:
                print(f'abnormal batch {batch_index} loss: {batch_loss.item() / len(batch)}, {num_spam}/{batch_size} spam')
                for i in batch:
                    abnormal_batches[epoch_index].append(int(i))

        # log the current learning rate
        learning_rates.append(learning_rate)

        # check testing loss AND full test and add to list
        test_metrics[0].append(find_loss(rnn, criterion, testing_data, batches))
        temp_test_metrics = test(rnn, testing_data, show_graph=False)
        for i in range(1, len(test_metrics)):
            test_metrics[i].append(temp_test_metrics[i-1])

        # save model every epoch
        torch.save(rnn, './my_model')

        # log the current training loss
        current_loss /= len(batches)
        train_metrics[0].append(current_loss)
        graph_total = confusion_matrix.sum()
        # get testing metrics
        if graph_total > 0:
            confusion_matrix /= graph_total
        percent_correct = float(confusion_matrix[0][0]+confusion_matrix[1][1])*100
        precision = float(confusion_matrix[1][1]/sum(confusion_matrix[1])) # AKA when you guess spam, how many were right
        recall = float(confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])) # AKA of all the spam, how many did you guess right
        f1_score = 2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        train_metrics[1].append(percent_correct)
        train_metrics[2].append(precision)
        train_metrics[3].append(recall)
        train_metrics[4].append(f1_score)

        # print training metrics and loss
        print(f'\nTRAINING METRICS:\n{confusion_matrix}')
        print(f'{percent_correct}% correct')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1 score: {f1_score}')
        print(f'\nFINISH EPOCH {epoch_index}: training average batch loss = {train_metrics[0][-1]}, testing loss = {test_metrics[0][-1]}\n')

        # cut early if you reach the goal
        if test_metrics[0][-1] < target_loss:
            break

    # show training results
    if show_graph:
        plt.figure()
        # plot all the training metrics
        for i in range(len(train_metrics)):
            if i != 1 and i != 2 and i != 3:
                plt.plot(train_metrics[i])

        # plot all the testing metrics
        for i in range(len(test_metrics)):
            if i != 1 and i != 2 and i != 3:
                plt.plot(test_metrics[i])

        # plot all the learning rates
        plt.plot(learning_rates)

        plt.legend(['train loss', 
                    # 'train accuracy', 'train precision', 'train recall',
                    'train f1',
                    'test loss', 
                    # 'test accuracy', 'test precision', 'test recall', 
                    'test f1',
                    'learning rates'])
        plt.show()

    if print_outlier_batches: # ONLY FOR FINE TUNING
        with open("abnormal_batches.txt", "w") as f:
            f.write(f'batches = {abnormal_batches}')
    print(f'train_metrics = {train_metrics}')
    print(f'test_metrics = {test_metrics}')
    print(f'learning_rates = {learning_rates}')
    torch.save(rnn, "./my_model")
    return train_metrics, test_metrics, learning_rates

# TUNE THE THRESHOLD FOR SPAM
def threshold_tuner(rnn, validation_data:torch.utils.data.Subset, lower_bound:float, 
                    upper_bound:float, num_steps:int, ham_percent:float) -> float:
    """
    lower_bound and upper_bound are both INCLUSIVE
    """

    step_size:float = round((upper_bound-lower_bound)/(num_steps-1), 4)
    best_f1 = 0
    best_threshold = 0

    indices_in_all_data = get_batches_from_dataset(validation_data, len(validation_data), ham_percent, num_batches=1)[0]

    cur_threshold = lower_bound
    while cur_threshold <= upper_bound:
        cur_f1 = test(rnn, validation_data, show_graph=False, threshold=cur_threshold, print_metrics=False, 
                      indices_in_all_data=indices_in_all_data)[3]
        print(f'F1 for threshold {cur_threshold}: {cur_f1}')
        if cur_f1 > best_f1:
            best_threshold = cur_threshold
            best_f1 = cur_f1

        cur_threshold = round(cur_threshold+step_size, 4)

    # save the threshold so it can be used later
    with open('best_threshold.txt', 'w') as file:
        file.write(str(best_threshold))

    return best_threshold

# TEST NEURAL NETWORK
def test(rnn, testing_data:torch.utils.data.Subset, show_graph:bool = True, threshold = 0.5, 
         print_metrics = True, indices_in_all_data:list[int] = []):
    """
    returns accuracy, precision, recall, f1 score
    """
    # turn testing_data into a list of indices in all_data to allow RUS to work
    if len(indices_in_all_data) == 0:
        indices_in_all_data = get_batches_from_dataset(testing_data, len(testing_data), 1., num_batches=1)[0]

    classes = testing_data.dataset.labels_unique

    confusion_matrix = torch.zeros(len(classes), len(classes))

    if print_metrics: print(f'\nStart testing on {len(indices_in_all_data)} examples with threshold {threshold}\n')

    rnn.eval() # turn on eval flag
    with torch.no_grad(): # don't record gradients
        for index, element in enumerate(indices_in_all_data): # go thru each test example
            (label_tensor, data_tensor, label, data) = testing_data.dataset[element]
            label_index = classes.index(label) # the index of the correct answer for this testcase

            output = rnn(data_tensor)
            guess, guess_index = postprocess.label_from_output(output, classes, threshold) # the guess and index of the guess

            if index % max(round(len(indices_in_all_data)/10),1) == 0: # print 10 outputs (max prevent /0 error)
                if print_metrics: print(f"testcase index in all_data: {element}, guess: {str(guess)}, guess_idx: {str(guess_index)}, label: {str(label)}, label_idx: {str(label_index)}")

            confusion_matrix[guess_index][label_index] += 1

    # turn the total count into percentage (aka normalize)
    graph_total = confusion_matrix.sum()
    if graph_total > 0:
        confusion_matrix /= graph_total
    percent_correct = float(confusion_matrix[0][0]+confusion_matrix[1][1])*100.
    precision = float(confusion_matrix[1][1]/sum(confusion_matrix[1])) if sum(confusion_matrix[1]) != 0 else 0 # AKA when you guess spam, how many were right
    recall = float(confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])) if (confusion_matrix[0][1]+confusion_matrix[1][1]) != 0 else 0 # AKA of all the spam, how many did you guess right
    f1_score = 2*precision*recall/(precision+recall) if precision+recall!=0 else 0

    if print_metrics:
        print(f'\nTESTING METRICS:')
        print(confusion_matrix)
        print(f'{percent_correct}% correct')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1 score: {f1_score}')

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

    return percent_correct, precision, recall, f1_score

def main():
    print(torch.__version__)

    # use GPU if possible
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    torch.set_default_device(device)

    print(device)

    # SETUP DATASET
    all_data = MyDataset([',', '\t', ','], 
                         ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection', # first 2 was 11147 total testcases
                          'data/Mendeley Data collection/Dataset_5971.csv']) # total roughly 17000 testcases
    train_set, validation_set, test_set = torch.utils.data.random_split(all_data, [.6, .2, .2], generator=torch.Generator(device=device))

    # CREATE/TRAIN NN
    from_scratch:bool = False # use a new model OR keep a previous model

    train_model:bool = True
    fine_adjustment:bool = False # make big steps OR fine adjustments

    do_tuning:bool = True
    
    test_model:bool = True

    # train from scratch or load a previous pretrained model
    if from_scratch:
        rnn = MyRNN_Mini_Boi(len(preprocess.allowed_char), 256, len(all_data.labels_unique))
    else:
        rnn = torch.load('./my_model', weights_only = False)

    rnn.to(device)
    print(rnn)

    criterion = nn.NLLLoss(weight = torch.tensor([1., 5.]))
    ham_percent = 0.15

    if train_model:
        # if making final adjustments to a model, use the custom dynamic lr param
        if fine_adjustment:
            num_epoch = 20
            target_loss = 0.
            num_batches = 8
            low_bound = 0.001*2**-4
            num_steps = 6
            epoch_per_dynamic_lr = 1
            target_progress_per_epoch = 1.0 # forces dynamic lr each epoch, regardless of improvement
            
            train(rnn, train_set, validation_set, test_set, ham_percent, num_epoch=num_epoch, target_loss=target_loss,
                  criterion=criterion, epoch_per_dynamic_lr=epoch_per_dynamic_lr, target_progress_per_epoch=target_progress_per_epoch, 
                  num_batches=num_batches, low_bound=low_bound, num_steps=num_steps, print_outlier_batches=False)
        # if making big steps, use the defaults
        else:
            train(rnn, train_set, validation_set, test_set, ham_percent, criterion=criterion)

    if do_tuning: best_threshold = threshold_tuner(rnn, validation_set, lower_bound=0.1, upper_bound=0.9, 
                                                   num_steps=17, ham_percent=1)
    else: best_threshold = 0.5

    if test_model:
        indices_in_all_data = get_batches_from_dataset(test_set, len(test_set), 1., num_batches=1)[0]
        test(rnn, test_set, show_graph=False, threshold=best_threshold, indices_in_all_data=indices_in_all_data)
        test(rnn, test_set, show_graph=False, threshold=0.5, indices_in_all_data=indices_in_all_data)

if __name__ == "__main__":
    main()