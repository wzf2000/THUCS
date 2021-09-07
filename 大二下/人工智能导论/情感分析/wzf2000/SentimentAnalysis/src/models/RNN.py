import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from .Model import Model
from ..modules.RNN import RNN, GRU, LSTM

class RNNModel(Model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Model.add_model_specific_args(parent_parser)
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of GRU layers.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size of GRU.')
        parser.add_argument('--RNN_type', type=str, default='GRU',
                            help='The type of RNN.')
        return parser

    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, dropout_rate: float = 0.5, num_layers: int = 2, hidden_size: int = 64, RNN_type: str = 'GRU', *args, **kwargs):
        super().__init__(word_num, label_num, feature_num, dropout_rate, *args, **kwargs)
        self.word_emb = torch.nn.Embedding(self.word_num + 1, self.feature_num)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.RNN_type = RNN_type
        self.rnn_layer = eval(self.RNN_type)(self.feature_num, self.hidden_size, self.num_layers, dropout = self.dropout_rate)
        self.linear = torch.nn.Linear(self.hidden_size, self.label_num)
        self.init_weights()

    def forward(self, sentence, valid, *args, **kwargs):
        emb = self.dropout(self.word_emb(sentence))  # B * len * v
        output, hidden = self.rnn_layer(emb, valid)  # layers * B * h
        rnn_vec = self.linear(hidden[-1])  # B * l
        return rnn_vec
    
    def get_name(self):
        return self.RNN_type + 'Model'

class RNNAttentionModel(Model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Model.add_model_specific_args(parent_parser)
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of GRU layers.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size of GRU.')
        parser.add_argument('--att_size', type=int, default=64,
                            help='Size of attention.')
        parser.add_argument('--RNN_type', type=str, default='GRU',
                            help='The type of RNN.')
        return parser

    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, dropout_rate: float = 0.5, num_layers: int = 2, hidden_size: int = 64, att_size: int = 64, RNN_type: str = 'GRU', *args, **kwargs):
        super().__init__(word_num, label_num, feature_num, dropout_rate, *args, **kwargs)
        self.word_emb = torch.nn.Embedding(self.word_num + 1, self.feature_num)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.RNN_type = RNN_type
        self.att_size = att_size
        self.rnn_layer = eval(self.RNN_type)(self.feature_num, self.hidden_size, self.num_layers, dropout = self.dropout_rate)
        self.att1 = torch.nn.Linear(self.hidden_size, self.att_size)
        self.att2 = torch.nn.Linear(self.hidden_size, self.att_size)
        self.att_v = torch.nn.Linear(self.att_size, 1)
        # self.W1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.W2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.U = torch.nn.Linear(self.hidden_size, 1)
        self.linear = torch.nn.Linear(self.hidden_size, self.label_num)
        self.init_weights()

    def forward(self, sentence, valid, *args, **kwargs):
        emb = self.dropout(self.word_emb(sentence))  # B * len * v
        output, hidden = self.rnn_layer(emb, valid)  # B * len * h, layers * B * h
        att1 = self.att1(hidden[-1])  # B * a
        att2 = self.att2(output)  # B * len * a
        alpha = self.att_v((att1.unsqueeze(dim = -2) + att2).sigmoid())  # B * len * 1
        c = (output * alpha * valid.unsqueeze(dim = -1)).sum(dim = -2)  # B * h
        # uo = torch.tanh(self.W1(output))  # B * len * h
        # uh = torch.tanh(self.W2(hidden[-1]))  # B * h
        # att_score = F.softmax((uo * uh.unsqueeze(dim = -2)).sum(dim = -1, keepdim = True), dim = 1)  # B * len * 1
        # c = (att_score * output).sum(dim = 1)  # B * h
        rnn_vec = self.linear(c)  # B * l
        return rnn_vec
    
    def get_name(self):
        return self.RNN_type + 'AttentionModel'