# -*- coding: utf-8 -*-
import numpy as np
import chainer, os, collections, six, math, random, time, copy, math, json, os, sys
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import sequential
from args import args

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class Params():
	def __init__(self, dict=None):
		self.num_classes = 10
		self.weight_std = 1
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

		if dict:
			self.from_dict(dict)

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			if hasattr(value, "to_dict"):
				dict[attr] = value.to_dict()
			else:
				dict[attr] = value
		return dict

	def dump(self):
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

class Discriminator():
	def __init__(self, params):
		self.params = copy.deepcopy(params)
		self.config = to_object(params["config"])

		self.discriminator = sequential.chain.Chain(self.config.weight_initializer, self.config.weight_std)
		self.discriminator.add_sequence(sequential.from_dict(params["model"]))
		self.discriminator.setup_optimizers(self.config.optimizer, self.config.learning_rate, self.config.momentum)

		self._gpu = False

	def to_gpu(self):
		self.discriminator.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def discriminate(self, x_batch, test=False, apply_softmax=True):
		x_batch = self.to_variable(x_batch)
		prob = self.discriminator(x_batch, test=test)
		if apply_softmax:
			prob = F.softmax(prob)
		return prob

	def backprop(self, loss):
		self.discriminator.backprop(loss)

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.discriminator.load(dir + "/model.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.discriminator.save(dir + "/model.hdf5")
