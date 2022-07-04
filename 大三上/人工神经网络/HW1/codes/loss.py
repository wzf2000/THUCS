from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        dis = input - target
        return 0.5 * (dis ** 2).sum(-1).mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        dis = input - target
        return dis / input.shape[0]
        # TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp_input = np.exp(input - input.max(-1, keepdims=True))
        prob = exp_input / (exp_input.sum(-1, keepdims=True))
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return (-np.log(prob) * target).sum(-1).mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        exp_input = np.exp(input - input.max(-1, keepdims=True))
        prob = exp_input / (exp_input.sum(-1, keepdims=True))
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return (prob - target) / input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START
        return (np.maximum((self.margin - (input * target).sum(-1, keepdims=True) + input), 0).sum(-1) - self.margin).mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        grad = ((self.margin - (input * target).sum(-1, keepdims=True) + input) > 0).astype(float)
        grad -= target * np.sum(grad, -1, keepdims=True)
        return grad / input.shape[0]
        # TODO END

