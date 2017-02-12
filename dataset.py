# -*- coding: utf-8 -*-
import os
import numpy as np
import mnist_tools

def load_train_images():
	return mnist_tools.load_train_images()

def load_test_images():
	return mnist_tools.load_test_images()

def binarize_data(x):
	threshold = np.random.uniform(size=x.shape)
	return np.where(threshold < x, 1.0, 0.0).astype(np.float32)

def split_data(images, labels, num_validation_data=10000, seed=0):
	training_x = []
	training_labels = []
	validation_x = []
	validation_labels = []

	np.random.seed(seed)
	indices = np.arange(len(images))
	np.random.shuffle(indices)

	for n in xrange(len(images)):
		index = indices[n]
		if len(validation_x) >= num_validation_data:
			training_x.append(images[index])
			training_labels.append(labels[index])
		else:
			validation_x.append(images[index])
			validation_labels.append(labels[index])

	return np.asarray(training_x, dtype=np.float32), np.asarray(training_labels, dtype=np.int32), np.asarray(validation_x, dtype=np.float32), np.asarray(validation_labels, dtype=np.int32)
	
def sample_data(images, labels, batchsize, binarize=True):
	ndim_x = images[0].size
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	image_batch = images[indices]
	label_batch = labels[indices]
	if binarize:
		image_batch = binarize_data(image_batch)
	# [0, 1] -> [-1, 1]
	image_batch = image_batch * 2.0 - 1.0
	return image_batch, label_batch