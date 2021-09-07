import torch
import pandas
import re
from tqdm import tqdm

class DataReader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.all_labels = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
        self.label_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'guilt': 3,
            'joy': 4,
            'sadness': 5,
            'shame': 6
        }
        self.label_num = 7

    def convert(self, datas: pandas.DataFrame):
        ids = datas['id'].tolist()
        labels = datas['label'].tolist()
        sentences = datas['sentence'].tolist()
        labels = [ self.label_map[label] for label in labels ]
        convert_sentences = []
        for sentence in tqdm(sentences):
            sentence = re.compile(r'\b[a-zA-Z]+\b', re.I).findall(sentence)
            sentence = [ self.words.index(word) + 1 for word in sentence ]
            convert_sentences.append(sentence)
        return {
            'id': ids,
            'label': labels,
            'sentence': convert_sentences
        }
    
    def count(self, sentences: list):
        self.words = set()
        for sentence in tqdm(sentences):
            sentence = re.compile(r'\b[a-zA-Z]+\b', re.I).findall(sentence)
            self.words = self.words.union(set(sentence))
        self.words = list(self.words)
        self.words.sort()

    def read_datas(self):
        self.train_datas = pandas.read_csv(self.data_dir + 'isear_train.csv', names = [
            'id', 'label', 'sentence'
        ], sep = ',', usecols = range(3), header = 0)
        self.validate_datas = pandas.read_csv(self.data_dir + 'isear_valid.csv', names = [
            'id', 'label', 'sentence'
        ], sep = ',', usecols = range(3), header = 0)
        self.test_datas = pandas.read_csv(self.data_dir + 'isear_test.csv', names = [
            'id', 'label', 'sentence'
        ], sep = ',', usecols = range(3), header = 0)
        sentences = self.train_datas['sentence'].tolist() + self.validate_datas['sentence'].tolist() + self.test_datas['sentence'].tolist()
        self.count(sentences)
        self.train_datas = self.convert(self.train_datas)
        self.validate_datas = self.convert(self.validate_datas)
        self.test_datas = self.convert(self.test_datas)
        self.word_num = len(self.words)
        print('\n\n\n------------------------\nCount Words: ', self.word_num, '------------------------\n\n\n', sep = '\n')

