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

def create_semisupervised(images, labels, num_validation_data=10000, num_types_of_label=10, seed=0):
	training_x = []
	training_labels = []
	validation_x = []
	validation_labels = []

	np.random.seed(seed)
	indices = np.arange(len(images))
	np.random.shuffle(indices)

	for n in xrange(len(images)):
		index = indices[n]
		if len(validation_x) > num_validation_data:
			training_x.append(images[index])
			training_labels.append(labels[index])
		else:
			validation_x.append(images[index])
			validation_labels.append(labels[index])

	return training_x, training_labels, validation_x, validation_labels
	
def sample_data(images, labels, batchsize, ndim_x, ndim_y, binarize=True):
	image_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	label_id_batch = np.zeros((batchsize,), dtype=np.int32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		img = images[data_index].astype(np.float32) / 255.0
		image_batch[j] = img.reshape((ndim_x,))
		label_id_batch[j] = labels[data_index]
	if binarize:
		image_batch = binarize_data(image_batch)
	# [0, 1] -> [-1, 1]
	image_batch = image_batch * 2.0 - 1.0
	return image_batch, label_id_batch