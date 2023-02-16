import numpy as np
import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
from functools import partial

data_cls = natsorted(glob('data/thoughtviz_eeg_data/*'))
cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
idx2cls  = {value:key for key, value in cls2idx.items()}

# def map_func(path):
# 	eeg, class_name, subject_name = np.load(path, allow_pickle=True)
# 	eeg = np.transpose(eeg, [1, 0])
# 	class_idx  = cls2idx[class_name]
# 	return tf.cast(eeg, dtype=tf.float32), tf.cast(class_idx, dtype=tf.int32)

# def load_complete_data(data_path, batch_size=16):
# 	dataset = tf.data.Dataset.list_files(data_path)
# 	dataset = dataset.map(lambda X: tf.numpy_function( partial(map_func), [X], [tf.float32, tf.int32,],), num_parallel_calls=tf.data.experimental.AUTOTUNE)
# 	dataset = dataset.shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
# 	return dataset

def preprocess_data(X, Y):
	X = tf.squeeze(X, axis=-1)
	max_val = tf.reduce_max(X)/2.0
	X = (X - max_val) / max_val
	X = tf.transpose(X, [1, 0])
	X = tf.cast(X, dtype=tf.float32)
	Y = tf.argmax(Y)
	return X, Y

def load_complete_data(X, Y, batch_size=16):	
	dataset = tf.data.Dataset.from_tensor_slices((X, Y)).map(preprocess_data).shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset