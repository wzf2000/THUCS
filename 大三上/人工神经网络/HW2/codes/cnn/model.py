# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm2d, self).__init__()
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
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			miu = input.mean([0, 2, 3])
			sigma2 = input.var([0, 2, 3])
			self.running_mean = 0.9 * self.running_mean + 0.1 * miu
			self.running_var = 0.9 * self.running_var + 0.1 * sigma2
		else:
			miu = self.running_mean
			sigma2 = self.running_var
		output = (input - miu[:, None, None]) / torch.sqrt(sigma2[:, None, None] + 1e-5)
		output = self.weight[:, None, None] * output + self.bias[:, None, None]
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			output = torch.bernoulli(torch.ones_like(input) * (1 - self.p)) * input
			output = output / (1 - self.p)
		else:
			output = input
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.seq = nn.Sequential(
			nn.Conv2d(3, 64, 5),
			BatchNorm2d(64),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(3, 2),
			nn.Conv2d(64, 64, 5),
			BatchNorm2d(64),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(3, 2)
		)
		self.fc = nn.Linear(4 * 4 * 64, 10)
		# self.seq1 = nn.Sequential(
		# 	nn.Conv2d(3, 256, 5),
		# 	BatchNorm2d(256),
		# 	nn.ReLU(),
		# 	nn.AvgPool2d(3, 2),
		# )
		# self.seq2 = nn.Sequential(
		# 	nn.Conv2d(256, 256, 1),
		# 	BatchNorm2d(256),
		# 	nn.ReLU(),
		# 	nn.Conv2d(256, 256, 3, padding = 1),
		# 	BatchNorm2d(256),
		# 	nn.ReLU(),
		# 	nn.Conv2d(256, 256, 1),
		# 	BatchNorm2d(256)
		# )
		# self.relu = nn.ReLU()
		# self.pool = nn.AvgPool2d(3, 2)
		# self.fc = nn.Linear(6 * 6 * 256, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.seq(x)
		logits = torch.reshape(logits, (logits.shape[0], -1))
		logits = self.fc(logits)
		# logits = self.seq1(x)
		# logits = self.pool(self.relu(logits + self.seq2(logits)))
		# logits = torch.reshape(logits, (logits.shape[0], -1))
		# logits = self.fc(logits)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
