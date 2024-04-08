import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
import skimage.io as skio
from skimage.transform import resize
from tifffile import imwrite
import numpy as np
import cv2
from save_figure import save_figure, save_figure_condition
import h5py
from functools import partial
import tensorflow_io as tfio
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn')

# def map_func(path, resolution):
# 	X = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
# 	X = tf.image.resize(X, (resolution, resolution))
# 	X = (tf.cast( X, dtype=tf.float32 ) - 127.5) / 127.5
# 	path = path.decode('utf-8')
# 	Y = clstoidx[path.split(os.path.sep)[-2]]
# 	# code  = np.zeros(shape=(max(clstoidx.values())+1,), dtype=np.float32)
# 	# code[Y] = 1
# 	Y    = tf.cast(code, dtype=tf.float32)
# 	path = np.choice(imgdict[Y], size=(1,), replace=True)
# 	return X, Y

def preprocess_data(X, Y, P, resolution=128):
	X = tf.squeeze(X, axis=-1)
	max_val = tf.reduce_max(X)/2.0
	X = (X - max_val) / max_val
	X = tf.transpose(X, [1, 0])
	X = tf.cast(X, dtype=tf.float32)
	Y = tf.argmax(Y)
	I = tf.image.decode_jpeg(tf.io.read_file(P), channels=3)
	I = tf.image.resize(I, (resolution, resolution))
	I = (tf.cast( I, dtype=tf.float32 ) - 127.5) / 127.5

	return X, Y, I

def load_complete_data(X, Y, P, batch_size=16, dataset_type='train'):	
	if dataset_type == 'train':
		dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).map(preprocess_data).shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	else:
		dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).map(preprocess_data).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset


def show_batch_images(X, save_path, Y=None):
	# Y = np.squeeze(tf.argmax(Y, axis=-1).numpy())
	X = np.clip( np.uint8( ((X.numpy() * 0.5) + 0.5) * 255 ), 0, 255)
	# X = X[:16]
	col = 4
	row = X.shape[0] // col
	# print(X.shape[0], Y.shape)
	for r in range(row):
	    for c in range(col):
	        plt.subplot2grid((row, col), (r, c), rowspan=1, colspan=1)
	        plt.grid('off')
	        plt.axis('off')
	        if Y is not None:
	        	plt.title('{}'.format(Y[r*col+c]))
	        plt.imshow(X[r*col+c])
	plt.tight_layout()
	plt.savefig(save_path)
	plt.clf()
	plt.close()