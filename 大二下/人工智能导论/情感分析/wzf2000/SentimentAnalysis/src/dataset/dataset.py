import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data : dict, word_num : int, *args, **kwargs):
        self.data = data
        self.word_num = word_num

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index: int):
        sentence = self.data['sentence'][index]
        label = self.data['label'][index]
        return (sentence, label)
