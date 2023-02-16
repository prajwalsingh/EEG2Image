# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
from glob import glob
from natsort import natsorted
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

model = InceptionV3(weights='tmp/imagenet/inception-2015-12-05.tgz')

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	global model
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for idx, i in enumerate(range(n_split), start=0):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		batch = tf.data.Dataset.from_tensor_slices((subset,)).batch(256)
		p_yx  = []
		for subset in tqdm(batch):
			subset = subset[0]
			p_y = model.predict(subset)
			p_yx.extend(p_y)
		p_yx = np.array(p_yx)
		print(p_yx.shape)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
		print('Inception score of class {}: {}'.format(idx, is_score))
		with open('experiments/thought_inceptionscore.txt', 'a') as file:
			file.write('Inception score of class {}: {}\n'.format(idx, is_score))
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

# # load cifar10 images
# (images, _), (_, _) = cifar10.load_data()
# # shuffle images
# shuffle(images)
# print('loaded', images.shape)
# calculate inception score
for path in natsorted(glob('experiments/inception/*')):
	images = []
	for im_path in tqdm(natsorted(glob(path+'/*'))):
		images.append(cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB))
	images = np.array(images)
	is_mean, is_std = calculate_inception_score(images)

	print('Inception score for epoch {}: ({}, {})'.format(os.path.split(path)[-1], is_mean, is_std))

	with open('experiments/thought_inceptionscore.txt', 'a') as file:
		file.write('-'*30+'\n')
		file.write('Inception score for epoch {}: ({}, {})\n'.format(os.path.split(path)[-1], is_mean, is_std))
		file.write('-'*30+'\n\n')