import torch
from torch import nn, sigmoid, tanh, Tensor
from math import sqrt


class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables
        k = sqrt(1 / hidden_size)

        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, (4 * m))
        # b, the vector of biases has size (4 * m)
        self.b = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))
        self.W = nn.Parameter(torch.empty(vocab_size, 4 * hidden_size).uniform_(-k, k))
        self.V = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size).uniform_(-k, k))

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """
        # pre activation has size (Bx4m)
        a = self.b + (x @ self.W) + (h @ self.V)

        # need to split a into a_i, a_f, a_o, a_g all of Bxm size
        a_i, a_f, a_o, a_g = torch.split(a, self.hidden_size, dim=1)
        i = sigmoid(a_i) #Bxm
        f = sigmoid(a_f) #Bxm
        o = sigmoid(a_o) #Bxm
        g = tanh(a_g)    #Bxm

        # calculate new hidden state h_out and new cell state c_out using point-wise operators
        c_out = i * g + f * c   #Bxm
        h_out = o * tanh(c_out) #Bxm

        return h_out, c_out
