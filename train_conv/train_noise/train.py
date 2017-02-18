import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from progress import Progress
from mnist_tools import load_train_images, load_test_images
from model import model
from args import args

def compute_accuracy(image_batch, label_batch):
	image_batch = np.reshape(image_batch, (-1, 1, 28, 28))
	num_data = image_batch.shape[0]
	images_l_segments = np.split(image_batch, num_data // 500)
	label_ids_l_segments = np.split(label_batch, num_data // 500)
	sum_accuracy = 0
	for image_batch, label_batch in zip(images_l_segments, label_ids_l_segments):
		distribution = model.discriminate(image_batch, apply_softmax=True, test=True)
		accuracy = F.accuracy(distribution, model.to_variable(label_batch))
		sum_accuracy += float(accuracy.data)
	return sum_accuracy / len(images_l_segments)

def main():
	# config
	config = model.config

	# load MNIST images
	num_data = 60000
	images = np.random.random(size=(num_data, 28 * 28)).astype(np.float32)
	labels = np.random.randint(0, config.num_classes, size=num_data).astype(np.int32)

	# settings
	max_epoch = 10000
	num_trains_per_epoch = 5000
	num_validation_data = 10000
	batchsize = 128

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# save validation accuracy per epoch
	csv_results = []

	# create semi-supervised split
	training_images, training_labels, validation_images, validation_labels = dataset.split_data(images, labels, num_validation_data, seed=args.seed)
	training_labels = np.random.randint(0, config.num_classes, training_labels.size).astype(np.int32)
	validation_labels = np.random.randint(0, config.num_classes, validation_labels.size).astype(np.int32)

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			image_batch, label_batch = dataset.sample_data(training_images, training_labels, batchsize, binarize=False)
			image_batch = np.reshape(image_batch, (-1, 1, 28, 28))
			distribution = model.discriminate(image_batch, apply_softmax=False)
			loss = F.softmax_cross_entropy(distribution, model.to_variable(label_batch))
			sum_loss += float(loss.data)

			model.backprop(loss)

			if t % 10 == 0:
				progress.show(t, num_trains_per_epoch, {})

		model.save(args.model_dir)
		train_accuracy = compute_accuracy(training_images, training_labels)
		validation_accuracy = compute_accuracy(validation_images, validation_labels)
		
		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"loss": sum_loss / num_trains_per_epoch,
			"accuracy (validation)": validation_accuracy,
			"accuracy (train)": train_accuracy,
		})

		# write accuracy to csv
		csv_results.append([epoch, train_accuracy, validation_accuracy, progress.get_total_time()])
		data = pd.DataFrame(csv_results)
		data.columns = ["epoch", "train_accuracy", "validation_accuracy", "min"]
		data.to_csv("{}/result.csv".format(args.model_dir))

if __name__ == "__main__":
	main()
