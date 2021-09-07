import torch
from argparse import ArgumentParser

class Model(torch.nn.Module):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--optimizer_name', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--l2', type=float, default=1e-6,
                            help='Weight of l2_regularize in pytorch optimizer.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--max_epoch', type=int, default=200,
                            help='Max epochs.')
        parser.add_argument('--es_patience', type=int, default=10,
                            help='#epochs with no improvement after which training will be stopped (early stop).')
        parser.add_argument('--feature_num', type=int, default=64,
                            help='Number of feature for word.')
        parser.add_argument('--dropout_rate', type=float, default=0.5,
                            help='Dropout rate.')
        return parser
    
    def __init__(self, word_num: int, label_num: int, feature_num: int = 64, dropout_rate: float = 0.5, *args, **kwargs):
        super().__init__()
        self.word_num = word_num
        self.label_num = label_num
        self.feature_num = feature_num
        self.dropout_rate = dropout_rate

    def init_weights(self) -> None:
        for n, p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, mean = 0, std = 0.01)
    
    def get_name(self):
        return 'Model'