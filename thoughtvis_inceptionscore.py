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
from eval_utils import get_inception_score

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

for path in natsorted(glob('experiments/inception/210')):
	images = []
	for im_path in tqdm(natsorted(glob(path+'/*'))):
		images.append(cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB))
	# images = np.array(images)
	is_mean, is_std = get_inception_score(images, splits=10)

	print('Inception score for epoch {}: ({}, {})'.format(os.path.split(path)[-1], is_mean, is_std))

	with open('experiments/thought_inceptionscore.txt', 'a') as file:
		file.write('-'*30+'\n')
		file.write('Inception score for epoch {}: ({}, {})\n'.format(os.path.split(path)[-1], is_mean, is_std))
		file.write('-'*30+'\n\n')