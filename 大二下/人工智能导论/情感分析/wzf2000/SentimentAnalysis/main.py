from src.data_reader.data_reader import DataReader
from src.dataset.dataset import Dataset
from src.models.CNN import ConvModel
from src.models.RNN import RNNModel, RNNAttentionModel
from src.models.MLP import MLPModel
from src.models.SelfAttention import SelfAttentionModel, SelfAttentionWithPosModel
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import time
import os
import random

SEED = 1949

parser_ori = ArgumentParser()
parser_ori.add_argument('--model', type=str, default='RNNModel',
                    help='Model name.')
parser_ori.add_argument('--test', type=str, default='No',
                    help='If just for test metrics.')
args_ori, _ = parser_ori.parse_known_args()
parser = ArgumentParser()
parser = eval(args_ori.model).add_model_specific_args(parser)
args, __ = parser.parse_known_args()

def collate_data(batch):
    max_len = max([ len(sentence) for sentence, label in batch ])
    new_sentences = []
    labels = []
    valid = []
    for sentence, label in batch:
        valid.append([1.] * len(sentence) + [0.] * (max_len - len(sentence)))
        sentence = sentence + [0] * (max_len - len(sentence))
        new_sentences.append(sentence)
        labels.append(label)
    batch = {}
    batch['sentence'] = torch.tensor(new_sentences)
    batch['label'] = torch.tensor(labels)
    batch['valid'] = torch.tensor(valid)
    return batch

def calc_metric(output, label):
    prediction = output.argmax(dim = 1, keepdim = False)
    return {
        'accuracy': ((prediction == label).float().sum(dim = 0, keepdim = False) / label.shape[0]).item(),
        'macro': f1_score(label.numpy().tolist(), prediction.numpy().tolist(), average='macro'),
        'micro': f1_score(label.numpy().tolist(), prediction.numpy().tolist(), average='micro')
    }

def test(model, data_loader, single: bool = True):
    model.eval()
    t = tqdm(data_loader)
    for batch in t:
        t.set_description('Epoch %i' % 0)
        optimizer.zero_grad()
        label = batch['label']
        output = model(batch['sentence'], valid = batch['valid'])
        if single:
            return calc_metric(output, label)['accuracy']
        else:
            return calc_metric(output, label)

def train(model, optimizer, data_loader, epoch, validate_loader):
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    t = tqdm(data_loader)
    for batch in t:
        t.set_description('Epoch %i' % epoch)
        optimizer.zero_grad()
        label = batch['label']
        output = model(batch['sentence'], valid = batch['valid'])
        loss = loss_func(output, label)
        t.set_postfix(loss = loss.item())
        loss.backward()
        optimizer.step()
    return test(model, validate_loader)

if __name__ == '__main__':
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    reader = DataReader('isear_v2/')
    reader.read_datas()
    train_dataset = Dataset(reader.train_datas, reader.word_num)
    validate_dataset = Dataset(reader.validate_datas, reader.word_num)
    test_dataset = Dataset(reader.test_datas, reader.word_num)
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_data)
    validate_loader = DataLoader(dataset = validate_dataset, batch_size = len(validate_dataset), collate_fn = collate_data)
    test_loader = DataLoader(dataset = test_dataset, batch_size = len(test_dataset), collate_fn = collate_data)

    model = eval(args_ori.model)(reader.word_num, reader.label_num, **vars(args))
    optimizer = eval('torch.optim.' + args.optimizer_name)(model.parameters(), lr = args.lr, weight_decay = args.l2)

    if not os.path.exists('save_models'):
        os.makedirs('save_models')
    if not os.path.exists('save_models/{}'.format(model.get_name())):
        os.makedirs('save_models/{}'.format(model.get_name()))

    if args.feature_num == 64:
        path_checkpoint = 'save_models/{}/checkpoint_{}_{}_best.pkl'.format(model.get_name(), args.lr, args.optimizer_name)
    else:
        path_checkpoint = 'save_models/{}/checkpoint_{}_{}_{}_best.pkl'.format(model.get_name(), args.lr, args.optimizer_name, args.feature_num)

    if not args_ori.test == 'Yes':
        best_metric = test(model, validate_loader)
        print('Val Metric:', best_metric)
        increase_cnt = 0

        for epoch in range(args.max_epoch):
            metric = train(model, optimizer, train_loader, epoch, validate_loader)
            if metric < best_metric:
                print('Val Metric:', metric)
                increase_cnt += 1
                if increase_cnt >= args.es_patience:
                    break
            else:
                print('Val Metric*:', metric)
                increase_cnt = 0
                best_metric = metric
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'metric': best_metric
                }
                torch.save(checkpoint, path_checkpoint)

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Epoch:', checkpoint['epoch'])
    print('Val Metric:', checkpoint['metric'])
    metric = test(model, test_loader, False)
    print('Test Metric:', metric)
