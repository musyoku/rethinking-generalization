# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, softmax

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
image_width = 28
image_height = image_width
ndim_latent_code = 50

# specify discriminator
discriminator_sequence_filename = args.model_dir + "/discriminator.json"

if os.path.isfile(discriminator_sequence_filename):
	print "loading", discriminator_sequence_filename
	with open(discriminator_sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(discriminator_sequence_filename))
else:
	config = Params()
	config.num_critic = 5
	config.weight_init_std = 0.001
	config.weight_initializer = "Normal"
	config.nonlinearity = "leaky_relu"
	config.optimizer = "rmsprop"
	config.learning_rate = 0.0001
	config.momentum = 0.9
	config.gradient_clipping = 1
	config.weight_decay = 0
	config.use_feature_matching = False
	config.use_minibatch_discrimination = False

	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(None, 500))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(500))
	model.add(Linear(None, 500))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(500))
	model.add(Linear(None, 10))

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(discriminator_sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

discriminator_params = params

gan = GAN(discriminator_params, generator_params)
gan.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	gan.to_gpu()