# -*- coding: utf-8 -*-
import sys, os, json
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from discriminator import Params, Discriminator
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, softmax

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

sequence_filename = args.model_dir + "/model.json"

if os.path.isfile(sequence_filename):
	print "loading", sequence_filename
	with open(sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(sequence_filename))
else:
	config = Params()
	config.num_classes = 10
	config.weight_init_std = 0.1
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.9
	config.gradient_clipping = 1
	config.weight_decay = 0

	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(None, 500))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(500))
	model.add(Linear(None, 500))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(500))
	model.add(Linear(None, config.num_classes))

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

model = Discriminator(params)
model.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	model.to_gpu()