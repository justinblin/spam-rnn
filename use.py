import torch
import preprocess
import postprocess
from rnn import MyRNN

def use(rnn:MyRNN, data:str, labels_unique:list[str]) -> tuple[str, int]:
    rnn.eval()
    with torch.no_grad():
        data_tensors = preprocess.string_to_tensor(data)
        output_tensor = rnn(data_tensors)
        guess, guess_index = postprocess.label_from_output(output_tensor, labels_unique)
        return guess, guess_index

# get the list of unique labels
labels_unique = ['ham', 'spam']

# load the trained model
rnn = torch.load('./my_model', weights_only = False)

# use the model
example_data = "Hey, you won a free Nitro subscription! Click here to claim your prize."
guess, guess_index = use(rnn, example_data, labels_unique)
print('My RNN guessed "' + example_data + '" is ' + guess)