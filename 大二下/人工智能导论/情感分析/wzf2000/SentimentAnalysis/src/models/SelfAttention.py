import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from .Model import Model

class SelfAttentionModel(Model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Model.add_model_specific_args(parent_parser)
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of SA Blocks.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size of self-attention.')
        parser.add_argument('--num_heads', type=int, default=1,
                            help='Number of heads in self-attention.')
        parser.add_argument('--att_size', type=int, default=64,
                            help='Size of attention.')
        return parser

    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, dropout_rate: float = 0.5, num_layers: int = 2, num_heads: int = 1, hidden_size: int = 64, att_size: int = 64, *args, **kwargs):
        super().__init__(word_num, label_num, feature_num, dropout_rate, *args, **kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_seq_len = 200
        self.word_emb = torch.nn.Embedding(self.word_num + 1, self.feature_num)
        self.att_size = att_size
        # position_encoding = np.array([
        #     [pos / np.power(10000, 2.0 * (j // 2) / self.feature_num) for j in range(self.feature_num)]
        #     for pos in range(self.max_seq_len)
        # ])
        # position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        # position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # position_encoding = torch.FloatTensor(position_encoding)
        self.position_encoding = torch.nn.Embedding(self.max_seq_len, self.feature_num)
        # self.position_encoding.weight = torch.nn.Parameter(position_encoding, requires_grad = False)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = self.feature_num, nhead = self.num_heads, dim_feedforward = self.hidden_size, dropout = self.dropout_rate,
            activation = 'gelu')
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.att = torch.nn.Linear(self.hidden_size, self.att_size)
        self.att_v = torch.nn.Linear(self.att_size, 1)
        self.linear = torch.nn.Linear(self.feature_num, self.label_num)
        self.register_buffer(
            'position',
            torch.tensor(range(self.max_seq_len)))
        self.init_weights()

    def forward(self, sentence, valid, *args, **kwargs):
        his_length = sentence.size(-1)
        seqs = self.word_emb(sentence) # + self.position_encoding(self.position[:his_length]).unsqueeze(dim = 0)  # B * len * v
        seqs *= valid.unsqueeze(dim = -1)  # B * len * v
        timeline_mask = ~valid.bool()  # B * len
        seqs = self.encoder(seqs.transpose(0, 1),  # len * B * v
                            src_key_padding_mask = timeline_mask)  # len * B * v
        # vectors = seqs[valid.byte().sum(dim = -1) - 1, torch.arange(seqs.size(1), device = seqs.device), :]  # B * v
        # vectors = (vectors * valid.sum(dim = -1, keepdims = True).gt(0))  # B * v
        att = self.att(seqs)  # l * B * a
        alpha = self.att_v(att.sigmoid()).permute(1, 0, 2)  # B * l * 1
        c = (seqs.permute(1, 0, 2) * alpha * valid.unsqueeze(dim = -1)).sum(dim = -2)  # B * v
        rnn_vec = self.linear(c)  # B * l
        return rnn_vec
    
    def get_name(self):
        return 'SelfAttentionWithPosModel'

class SelfAttentionWithPosModel(Model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Model.add_model_specific_args(parent_parser)
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of SA Blocks.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size of self-attention.')
        parser.add_argument('--num_heads', type=int, default=1,
                            help='Number of heads in self-attention.')
        parser.add_argument('--att_size', type=int, default=64,
                            help='Size of attention.')
        return parser

    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, dropout_rate: float = 0.5, num_layers: int = 2, num_heads: int = 1, hidden_size: int = 64, att_size: int = 64, *args, **kwargs):
        super().__init__(word_num, label_num, feature_num, dropout_rate, *args, **kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_seq_len = 200
        self.word_emb = torch.nn.Embedding(self.word_num + 1, self.feature_num)
        self.att_size = att_size
        # position_encoding = np.array([
        #     [pos / np.power(10000, 2.0 * (j // 2) / self.feature_num) for j in range(self.feature_num)]
        #     for pos in range(self.max_seq_len)
        # ])
        # position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        # position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # position_encoding = torch.FloatTensor(position_encoding)
        self.position_encoding = torch.nn.Embedding(self.max_seq_len, self.feature_num)
        # self.position_encoding.weight = torch.nn.Parameter(position_encoding, requires_grad = False)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = self.feature_num, nhead = self.num_heads, dim_feedforward = self.hidden_size, dropout = self.dropout_rate,
            activation = 'gelu')
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.att = torch.nn.Linear(self.hidden_size, self.att_size)
        self.att_v = torch.nn.Linear(self.att_size, 1)
        self.linear = torch.nn.Linear(self.feature_num, self.label_num)
        self.register_buffer(
            'position',
            torch.tensor(range(self.max_seq_len)))
        self.init_weights()

    def forward(self, sentence, valid, *args, **kwargs):
        his_length = sentence.size(-1)
        seqs = self.word_emb(sentence) + self.position_encoding(self.position[:his_length]).unsqueeze(dim = 0)  # B * len * v
        seqs *= valid.unsqueeze(dim = -1)  # B * len * v
        timeline_mask = ~valid.bool()  # B * len
        seqs = self.encoder(seqs.transpose(0, 1),  # len * B * v
                            src_key_padding_mask = timeline_mask)  # len * B * v
        # vectors = seqs[valid.byte().sum(dim = -1) - 1, torch.arange(seqs.size(1), device = seqs.device), :]  # B * v
        # vectors = (vectors * valid.sum(dim = -1, keepdims = True).gt(0))  # B * v
        att = self.att(seqs)  # l * B * a
        alpha = self.att_v(att.sigmoid()).permute(1, 0, 2)  # B * l * 1
        c = (seqs.permute(1, 0, 2) * alpha * valid.unsqueeze(dim = -1)).sum(dim = -2)  # B * v
        rnn_vec = self.linear(c)  # B * l
        return rnn_vec
    
    def get_name(self):
        return 'SelfAttentionWithPosModel'
