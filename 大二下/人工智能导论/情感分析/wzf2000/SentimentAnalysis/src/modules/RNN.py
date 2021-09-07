import torch

class RNN(torch.nn.Module):
    def __init__(self, vec_size: int = 64, hidden_size: int = 64, num_layers: int = 1,
                 bias: bool = False, dropout: float = 0, bidirectional: bool = False):
        super().__init__()
        self.vec_size = vec_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.init_modules()

    def init_modules(self):
        self.rnn = torch.nn.RNN(
            input_size=self.vec_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional,
            batch_first=True)

    def initial_state(self, init, sort_idx):
        '''
        :param init: (num_layers*num_directions) * b * h
        :param sort_idx: b
        :return:
        '''
        if init is not None:
            init = init.index_select(dim=1, index=sort_idx)
            return init
        return init

    def forward(self, seq_vectors, valid, init=None):
        '''
        :param seq_vectors: b * l * v
        :param valid: b * l
        :param init: (num_layers*num_directions) * b * h
        :return:
        '''
        seq_lengths = valid.sum(dim=-1)  # b
        n_samples = seq_lengths.size()[0]
        seq_lengths_valid = seq_lengths.gt(0).float().unsqueeze(dim=0).unsqueeze(dim=-1)  # 1 * b * 1
        seq_lengths_clamped = seq_lengths.clamp(min=1)  # b

        # Sort
        sort_seq_lengths, sort_idx = torch.topk(seq_lengths_clamped, k=n_samples)  # b
        sort_seq_vectors = seq_vectors.index_select(dim=0, index=sort_idx)  # b * l * v

        # Pack
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq_vectors, sort_seq_lengths.cpu(), batch_first=True)

        # RNN
        sort_output, sort_hidden = self.rnn(seq_packed, self.initial_state(init, sort_idx))
        sort_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sort_output, batch_first=True, total_length=valid.size()[1])  # b * l * h/2h

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=n_samples, largest=False)[1]  # b
        output = sort_output.index_select(dim=0, index=unsort_idx) * valid.unsqueeze(dim=-1).float()  # b * l * h/2h
        hidden = sort_hidden.index_select(dim=1, index=unsort_idx) * seq_lengths_valid  # (num_layers * 1/2) * b * h
        return output, hidden


class GRU(RNN):
    def init_modules(self):
        self.rnn = torch.nn.GRU(
            input_size=self.vec_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional,
            batch_first=True)


class LSTM(RNN):
    def init_modules(self):
        self.rnn = torch.nn.LSTM(
            input_size=self.vec_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional,
            batch_first=True)

    def initial_state(self, init, sort_idx):
        '''
        :param init: [(num_layers*num_directions) * b * h] * 2
        :param sort_idx: b
        :return:
        '''
        if init is not None:
            init0 = init[0].index_select(dim=1, index=sort_idx)
            init1 = init[1].index_select(dim=1, index=sort_idx)
            return [init0, init1]
        return init
    def forward(self, seq_vectors, valid, init=None):
        '''
        :param seq_vectors: b * l * v
        :param valid: b * l
        :param init: (num_layers*num_directions) * b * h
        :return:
        '''
        seq_lengths = valid.sum(dim=-1)  # b
        n_samples = seq_lengths.size()[0]
        seq_lengths_valid = seq_lengths.gt(0).float().unsqueeze(dim=0).unsqueeze(dim=-1)  # 1 * b * 1
        seq_lengths_clamped = seq_lengths.clamp(min=1)  # b

        # Sort
        sort_seq_lengths, sort_idx = torch.topk(seq_lengths_clamped, k=n_samples)  # b
        sort_seq_vectors = seq_vectors.index_select(dim=0, index=sort_idx)  # b * l * v

        # Pack
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq_vectors, sort_seq_lengths.cpu(), batch_first=True)

        # RNN
        sort_output, (sort_hidden, __) = self.rnn(seq_packed, self.initial_state(init, sort_idx))
        sort_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sort_output, batch_first=True, total_length=valid.size()[1])  # b * l * h/2h

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=n_samples, largest=False)[1]  # b
        output = sort_output.index_select(dim=0, index=unsort_idx) * valid.unsqueeze(dim=-1).float()  # b * l * h/2h
        hidden = sort_hidden.index_select(dim=1, index=unsort_idx) * seq_lengths_valid  # (num_layers * 1/2) * b * h
        return output, hidden
