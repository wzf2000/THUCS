from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
# model.add(Linear('fc1', 784, 10, 0.01))
# model.add(Linear('fc1', 784, 64, 0.01))
# model.add(Relu('relu'))
# model.add(Gelu('gelu'))
# model.add(Sigmoid('sigmoid'))
# model.add(Linear('fc2', 64, 10, 0.01))
model.add(Linear('fc1', 784, 256, 0.01))
model.add(Gelu('gelu'))
model.add(Linear('fc2', 256, 64, 0.01))
model.add(Gelu('gelu'))
model.add(Linear('fc3', 64, 10, 0.01))

# loss = EuclideanLoss(name='loss')
loss = SoftmaxCrossEntropyLoss(name='loss')
# loss = HingeLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'layer': 'linear_gelu_linear_gelu_linear_SoftmaxCrossEntropy',
    # 'layer': 'linear_gelu_linear_SoftmaxCrossEntropy',
    'learning_rate': 0.01,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}


train_loss = []
test_loss = []
train_acc = []
test_acc = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    _, __ = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    train_loss.append(_)
    train_acc.append(__)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        _, __ = test_net(model, loss, test_data, test_label, config['batch_size'])
        test_loss.append(_)
        test_acc.append(__)

with open('../final_results.txt', mode='a') as f:
    f.write('__'.join([key + '=' + str(value) for key, value in config.items()]))
    f.write(', ')
    f.write(str(float(train_loss[-1])))
    f.write(', ')
    f.write(str(float(test_loss[-1])))
    f.write(', ')
    f.write(str(float(train_acc[-1])))
    f.write(', ')
    f.write(str(float(test_acc[-1])))
    f.write('\n')

import matplotlib.pyplot as plt

plt.figure(figsize=(13, 6))
plt.subplot(1, 2, 1)
plt.ylabel(r'Loss')
plt.xlabel(r'Epochs')
plt.plot(range(config['max_epoch']), train_loss, label="train loss")
plt.plot(range(config['max_epoch']), test_loss, label="test loss")
plt.legend()


plt.subplot(1, 2, 2)
plt.ylabel(r'Accuracy/$\%$')
plt.xlabel(r'Epochs')
plt.plot(range(config['max_epoch']), train_acc, label="train accuracy")
plt.plot(range(config['max_epoch']), test_acc, label="test accuracy")
plt.legend()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('../' + '__'.join([key + '=' + str(value) for key, value in config.items()]) + '.png')
plt.show()
