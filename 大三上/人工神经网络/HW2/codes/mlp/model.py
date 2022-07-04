# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.empty(num_features))
		self.bias = Parameter(torch.empty(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			miu = input.mean(0)
			sigma2 = input.var(0)
			self.running_mean = 0.9 * self.running_mean + 0.1 * miu
			self.running_var = 0.9 * self.running_var + 0.1 * sigma2
		else:
			miu = self.running_mean
			sigma2 = self.running_var
		output = (input - miu) / torch.sqrt(sigma2 + 1e-5)
		output = self.weight * output + self.bias
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			output = torch.bernoulli(torch.ones_like(input) * (1 - self.p)) * input
			output = output / (1 - self.p)
		else:
			output = input
		return output
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.classify = nn.Sequential(
			nn.Linear(3 * 32 * 32, 512),
			BatchNorm1d(512),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.Linear(512, 10)
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.classify(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
