import torch.nn as nn
import torch.nn.functional as F
import torch

# CREATE NN
class MyRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(MyRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size) # input to hidden size
        self.hidden_to_hidden_0 = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size) 
        self.softmax = nn.Softmax(dim = 1)
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.leaky_relu = nn.LeakyReLU()

        # self.hidden_layers = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size), # applies the linear equation (weights and biases)
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Linear(hidden_size, output_size),
        #     nn.LogSoftmax(dim=1) # does log soft max (kinda like relu but different way)
        # )

    def forward(self, line_tensor:torch.Tensor, show = False) -> torch.Tensor:
        rnn_out, hidden = self.rnn(line_tensor) # input to hidden layer

        output = self.hidden_to_hidden_0(hidden[0]) # hidden layer to hidden
        output = self.leaky_relu(output)
        if show: print(output)

        output = self.hidden_to_hidden_1(output) # hidden layer to hidden
        output = self.leaky_relu(output)
        if show: print(output)

        output = self.hidden_to_output(output) # hidden layer to output
        if show: print(output)
        output = self.leaky_relu(output)
        if show: print(output)
        if show: print(self.softmax(output))
        output = self.log_softmax(output) # does softmax and then ln so NLLLoss doesn't need it's own log (AKA faster)
            # softmax normalizes values 0-1 and sum to 1
        if show: print(output)

        # output = self.hidden_layers(hidden[0])

        return output