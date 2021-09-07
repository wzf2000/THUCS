import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from .Model import Model

class ConvModel(Model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Model.add_model_specific_args(parent_parser)
        parser.add_argument('--out_channels', type=int, default=100,
                            help='Out channels of CNN layer.')
        parser.add_argument('--filters', type=str, default='[3, 4, 5]',
                            help='Kernal size for each filter.')
        return parser

    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, out_chanels: int = 100, dropout_rate: float = 0.5, filters: str = '[3, 4, 5]', *args, **kwargs):
        super().__init__(word_num, label_num, feature_num, dropout_rate, *args, **kwargs)
        self.out_channels = out_chanels
        self.filters = eval(filters)
        self.word_emb = torch.nn.Embedding(self.word_num + 1, self.feature_num)
        self.cnn_layers = torch.nn.ModuleList()
        for filter in self.filters:
            self.cnn_layers.append(torch.nn.Conv1d(in_channels = self.feature_num, out_channels = self.out_channels, kernel_size = filter))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.linear = torch.nn.Linear(len(self.filters) * self.out_channels, self.label_num)
        self.init_weights()
    
    def forward(self, sentence, *args, **kwargs):
        emb = self.word_emb(sentence)  # B * len * v
        emb = emb.permute(0, 2, 1)  # B * v * len
        conved = [self.relu(layer(emb)) for layer in self.cnn_layers]  # nf * B * o * (l - f + 1)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # nf * B * o
        output = self.dropout(torch.cat(pooled, dim = 1))  # B * (nf * o)
        output = self.linear(output)  # B * l
        return output
    
    def get_name(self):
        return 'ConvModel'
