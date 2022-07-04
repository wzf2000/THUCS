import torch
from torch import nn
import torch.nn.functional as F

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        #return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        # flag indicates whether the position is valid. 1 for valid, 0 for invalid.
        output = (self.input_layer(incoming) + self.hidden_layer(state)).tanh()
        new_state = output # stored for next step
        return output, new_state

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.input_r = nn.Linear(input_size, hidden_size, bias=False)
        self.input_z = nn.Linear(input_size, hidden_size, bias=False)
        self.input_n = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_n = nn.Linear(hidden_size, hidden_size, bias=False)

        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        r = F.sigmoid(self.input_r(incoming) + self.hidden_r(state))
        z = F.sigmoid(self.input_z(incoming) + self.hidden_z(state))
        n = F.tanh(self.input_n(incoming) + r * self.hidden_n(state))
        output = (1 - z) * n + z * state
        new_state = output
        return output, new_state
        # TODO END

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.input_i = nn.Linear(self.input_size, self.hidden_size)
        self.input_f = nn.Linear(self.input_size, self.hidden_size)
        self.input_g = nn.Linear(self.input_size, self.hidden_size)
        self.input_o = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hidden_f = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hideen_g = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hidden_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state (which can be a tuple)
        return  (torch.zeros(batch_size, self.hidden_size, device=device), torch.zeros(batch_size, self.hidden_size, device=device))
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        h, c = state
        i = F.sigmoid(self.input_i(incoming) + self.hidden_i(h))
        f = F.sigmoid(self.input_f(incoming) + self.hidden_f(h))
        g = F.tanh(self.input_g(incoming) + self.hideen_g(h))
        o = F.sigmoid(self.input_o(incoming) + self.hidden_o(h))
        new_c = f * c + i * g
        new_h = o * F.tanh(new_c)
        output = o
        return output, (new_h, new_c)
        # TODO END