import torch
from lstm_cell import LSTMCell
from basic_rnn_cell import BasicRNNCell
from torch import nn, zeros, empty_like


class CustomRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        """
        Creates an recurrent neural network of type {basic_rnn, lstm_rnn}

        basic_rnn is an rnn whose layers implement a tanH activation function
        lstm_rnn is ann rnn whose layers implement an LSTM cell

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in each layer of the RNN.
        num_layers: (int), the number of RNN layers at each time step
        rnn_type: (string), the desired rnn type. rnn_type is a member of {'basic_rnn', 'lstm_rnn'}
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # create a ModuleList self.rnn to hold the layers of the RNN
        # and append the appropriate RNN layers to it
        self.rnn = nn.ModuleList()
        for i in range(num_layers):
            if rnn_type == "basic_rnn":
                self.rnn.append(BasicRNNCell(vocab_size, hidden_size))
            elif rnn_type == "lstm_rnn":
                self.rnn.append(LSTMCell(vocab_size, hidden_size))
            else:
                raise ValueError(f"Unknown RNN type {rnn_type}")
            vocab_size = hidden_size
        

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x the mini-batch of input sequence
        h: (Tensor) of size (l x B x m) where l is the number of layers and m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (l x B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the final hidden state of each time step in order
        h: (Tensor) of size (l x B x m), the hidden state of the last time step
        c: (Tensor) of size (l x B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """

        # compute the hidden states and cell states (for an lstm_rnn) for each mini-batch in the sequence
        B, T, n = x.shape

        # stores the output at each time step
        outs_list = []

        # loop over every time step
        for t in range(T):

            # input at time t, Bxn matrix
            x_t = x[:, t, :]

            # store the current hidden states and cell state at t, lxBxM
            new_h = []
            new_c = []

            # for each layer in the RNN at time t
            for l in range(self.num_layers):
                if self.rnn_type == "basic_rnn":

                    # new hidden state from the forward function
                    # taking input at t and previous hidden state
                    h_out = self.rnn[l](x_t, h[l])
                    new_h.append(h_out)

                    # the output of the layer is the same as the next hidden state
                    x_t = h_out

                elif self.rnn_type == "lstm_rnn":

                    # calculate the next hidden state
                    h_out, c_out = self.rnn[l](x_t, h[l], c[l])
                    new_h.append(h_out)
                    new_c.append(c_out)

                    # out put of current cell is the next hidden later
                    x_t = h_out

            # update inputed hidden states for next time steps
            h = torch.stack(new_h, dim=0)
            if self.rnn_type == "lstm_rnn":
                c = torch.stack(new_c, dim=0)

            outs_list.append(x_t)

        outs = torch.stack(outs_list, dim=1)

        return outs, h, c
