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
from rnn import MyRNN, MyRNN_4x_Linear_LeakyReLU
from learning_rate_finder import find_best_lr, find_loss

# train neural network
def train(rnn, training_data:torch.utils.data.Subset, testing_data:torch.utils.data.Subset, ham_percent:float, 
          num_epoch:int = 20, batch_size:int = 64, target_loss:float = 0.08, learning_rate:float = 0.064, 
          criterion = nn.NLLLoss(), show_graph:bool = True, epoch_per_dynamic_lr:int = 3, target_progress_per_epoch:float=0.03, 
          num_batches:int = 8, low_bound:float = 0.001, num_steps:int = 10, print_outlier_batches=False) -> tuple[list[float]]:
    # track loss over time
    train_losses = []
    test_metrics:tuple[list[float]] = ([],[],[],[],[]) # tuple of lists for loss, accuracy, precision, recall, and f1
    learning_rates = []
    abnormal_batches:list[list[int]] = []

    print(f'\nStart training on {len(training_data)} examples\n')

    # go thru each epoch
    for epoch_index in range(num_epoch):
        rnn.train() # flag that you're starting to train now (redo it each epoch bcs testing makes it go away)

        batches = get_batches_from_dataset(training_data, batch_size, ham_percent) # use the same batches for dynamic lr as training

        # look for a new lr if there's a loss plateau
        # check the loss every 3 epochs (default), if it isn't >=10% (default) better than the last time, find a new lr
        if epoch_per_dynamic_lr != 0 and epoch_index % epoch_per_dynamic_lr == 0:
            # exceptions for first and second activations since plateau detection needs at least 2 points
            if epoch_index == 0 or epoch_index == epoch_per_dynamic_lr or \
            train_losses[-1] > train_losses[-1-epoch_per_dynamic_lr]*(1 - target_progress_per_epoch*epoch_per_dynamic_lr):
                learning_rate = find_best_lr(rnn, criterion, training_data, ham_percent, batches, batch_size=batch_size, 
                                             num_batches=num_batches, low_bound=low_bound, num_steps=num_steps)
                
        print(f'start epoch {epoch_index}, learning rate: {learning_rate}')

        optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate, momentum = 0.5) # stochastic gradient descent
            # momentum uses previous steps in the current step, faster training by reducing oscillation

        current_loss = 0 # reset loss so it doesn't build up in the tracking

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
        temp_test_metrics = test(rnn, testing_data, testing_data.dataset.labels_unique, show_graph=False)
        for i in range(1, len(test_metrics)):
            test_metrics[i].append(temp_test_metrics[i-1])

        # save model every epoch
        torch.save(rnn, './my_model')

        # log the current training loss
        current_loss /= len(batches)
        train_losses.append(current_loss)
        print(f'\nFINISH EPOCH {epoch_index}: training average batch loss = {train_losses[-1]}, testing loss = {test_metrics[0][-1]}\n')

        # cut early if you reach the goal
        if test_metrics[0][-1] < target_loss:
            break

    # show training results
    if show_graph:
        plt.figure()
        plt.plot(train_losses)
        for i in range(len(test_metrics)):
            if i != 1 and i != 2 and i != 3:
                plt.plot(test_metrics[i])
        plt.plot(learning_rates)
        plt.legend(['train loss', 'test loss', 
                    # 'test accuracy', 'test precision', 'test recall', 
                    'test f1','learning rates'])
        plt.show()

    if print_outlier_batches: # ONLY FOR FINE TUNING
        with open("abnormal_batches.txt", "w") as f:
            f.write(f'batches = {abnormal_batches}')
    print(f'train_losses = {train_losses}')
    print(f'test_metrics = {test_metrics}')
    print(f'learning_rates = {learning_rates}')
    torch.save(rnn, "./my_model")
    return train_losses, test_metrics, learning_rates

# TUNE THE THRESHOLD FOR SPAM
def threshold_tuner(rnn:MyRNN_4x_Linear_LeakyReLU, training_data:torch.utils.data.Subset) -> float:
    rnn.eval()
    val_targets = []
    val_probs = []

    print(f'\nStart tuning\n')

    with torch.no_grad():
        for curr_example in training_data:
            val_targets.append(curr_example[0].cpu().item())
            probs = rnn.forward(curr_example[1])
            val_probs.append(round(math.e ** probs.cpu().tolist()[0][1], 4))

    precision, recall, thresholds = precision_recall_curve(
        np.array(val_targets), 
        np.array(val_probs)
    )
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Optimal F1 threshold: {optimal_threshold}")
    return optimal_threshold

# TEST NEURAL NETWORK
def test(rnn, testing_data:MyDataset, classes:list[str], show_graph:bool = True, threshold = 0.5):
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
            guess, guess_index = postprocess.label_from_output(output, classes, threshold) # the guess and index of the guess

            if index % max(round(len(testing_data)/10),1) == 0: # print 10 outputs (max prevent /0 error)
                print(f"testcase index: {index}, guess: {str(guess)}, guess_idx: {str(guess_index)}, label: {str(label)}, label_idx: {str(label_index)}")

            confusion_matrix[guess_index][label_index] += 1
            if guess_index == label_index:
                percent_correct += 1

    # turn the total count into percentage (aka normalize)
    graph_total = confusion_matrix.sum()
    if graph_total > 0:
        confusion_matrix /= graph_total
    percent_correct = (percent_correct*100)/len(testing_data)
    precision = float(confusion_matrix[1][1]/sum(confusion_matrix[1])) # AKA when you guess spam, how many were right
    recall = float(confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])) # AKA of all the spam, how many did you guess right
    f1_score = 2*precision*recall/(precision+recall)

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
    all_data = MyDataset([',', '\t'], ['data/kaggle spam.csv', 'data/UC Irvine collection/SMSSpamCollection']) # 11147 total testcases
    train_set, test_set, extra_set = torch.utils.data.random_split(all_data, [.8, .2, .0], generator=torch.Generator(device=device))

    # CREATE/TRAIN NN
    from_scratch:bool = False # use a new model OR keep a previous model

    train_model:bool = True
    fine_adjustment:bool = True # make big steps OR fine adjustments

    do_tuning:bool = True
    
    test_model:bool = True

    # train from scratch or load a previous pretrained model
    if from_scratch:
        rnn = MyRNN_4x_Linear_LeakyReLU(len(preprocess.allowed_char), 750, len(all_data.labels_unique))
    else:
        rnn = torch.load('./my_model', weights_only = False)

    rnn.to(device)
    print(rnn)

    criterion = nn.NLLLoss(weight = torch.tensor([1., 10.]))
    ham_percent = 0.075

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
            
            train(rnn, train_set, test_set, ham_percent, num_epoch=num_epoch, target_loss=target_loss,
                  criterion=criterion, epoch_per_dynamic_lr=epoch_per_dynamic_lr, target_progress_per_epoch=target_progress_per_epoch, 
                  num_batches=num_batches, low_bound=low_bound, num_steps=num_steps, print_outlier_batches=False)
        # if making big steps, use the defaults
        else:
            train(rnn, train_set, test_set, ham_percent, criterion=criterion)

    if do_tuning: best_threshold = threshold_tuner(rnn, train_set)
    else: best_threshold = 0.5

    if test_model: test(rnn, test_set, all_data.labels_unique, show_graph=False, threshold=best_threshold)

if __name__ == "__main__":
    main()