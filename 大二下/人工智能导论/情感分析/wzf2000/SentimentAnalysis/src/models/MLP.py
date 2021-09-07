import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from .Model import Model

class MLPModel(Model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Model.add_model_specific_args(parent_parser)
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size of MLP.')
        return parser

    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, dropout_rate: float = 0.5, hidden_size: int = 64, *args, **kwargs):
        super().__init__(word_num, label_num, feature_num, dropout_rate, *args, **kwargs)
        self.word_emb = torch.nn.Embedding(self.word_num + 1, self.feature_num)
        self.hidden_size = hidden_size
        self.relu = torch.nn.ReLU()
        self.mlp_layer = torch.nn.Linear(self.feature_num, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.label_num)
        self.init_weights()

    def forward(self, sentence, valid, *args, **kwargs):
        emb = self.word_emb(sentence)  # B * len * v
        output = self.relu(self.mlp_layer(emb)).permute(0, 2, 1)  # B * h * len
        output = F.max_pool1d(output, output.shape[2]).squeeze(2)  # B * h
        return self.linear(output)
    
    def get_name(self):
        return 'MLPModel'