import torch.nn as nn
import torch.nn.functional as F
import torch

# CREATE NN
class MyRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(MyRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size) # applies the linear equation (weights and biases)
        self.softmax = nn.LogSoftmax(dim = 1) # does log soft max (kinda like relu but different way)

    def forward(self, line_tensor:torch.Tensor) -> torch.Tensor:
        rnn_out, hidden = self.rnn(line_tensor) # input to hidden layer
        output = self.hidden_to_output(hidden[0]) # hidden layer to output
        output = self.softmax(output) # apply soft max to output

        return output