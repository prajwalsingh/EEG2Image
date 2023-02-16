import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd

style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

if __name__ == '__main__':

	data_cls = natsorted(glob('data/thoughtviz_eeg_data/*'))
	cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
	idx2cls  = {value:key for key, value in cls2idx.items()}

	test_batch_size = 1
	train_batch_size = 256
	n_classes  = 10

	train_batch = load_complete_data('data/thoughtviz_eeg_data/*/train/*', batch_size=train_batch_size)
	# val_batch   = load_complete_data('data/thoughtviz_image_data/*/val/*', cls2idx=cls2idx, batch_size=batch_size)
	test_batch   = load_complete_data('data/thoughtviz_image_data/*/test/*', batch_size=test_batch_size)
	exp_dir     = 'experiments/'

	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/triplenet_ckpt', max_to_keep=20)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint).expect_partial()
	START = int(triplenet_ckpt.step) // len(train_batch)
	if triplenet_ckptman.latest_checkpoint:
		print('Restored from the latest checkpoint, epoch: {}'.format(START))
	EPOCHS = 10
	cfreq  = 2 # Checkpoint frequency

	# test_loss = tf.keras.metrics.Mean()
	# test_acc  = tf.keras.metrics.SparseCategoricalAccuracy()
	# tq = tqdm(test_batch)
	# for idx, (X, Y) in enumerate(tq, start=1):
	# 	loss = test_step(triplenet, X, Y)
	# 	test_loss.update_state(loss)
	# 	Y_cap, _ =triplenet(X, training=False)
	# 	test_acc.update_state(Y, )
	# 	tq.set_description('Test Loss: {}, Acc: {}'.format(test_loss.result(), test_acc.result()))

	test_loss = tf.keras.metrics.Mean()
	test_acc  = tf.keras.metrics.SparseCategoricalAccuracy()
	tq = tqdm(test_batch)
	feat_X = []
	feat_Y = []
	for idx, (X, Y) in enumerate(tq, start=1):
		feat = triplenet(X, training=False)
		feat_X.extend(feat.numpy())
		feat_Y.extend(Y.numpy())
	feat_X = np.array(feat_X)
	feat_Y = np.array(feat_Y)
	print(feat_X.shape, feat_Y.shape)
	# colors = list(plt.cm.get_cmap('viridis', 10))
	# print(colors)
	# colors  = [np.random.rand(3,) for _ in range(10)]
	# print(colors)
	# Y_color = [colors[label] for label in feat_Y]

	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
	tsne_results = tsne.fit_transform(feat_X)
	df = pd.DataFrame()
	df['label'] = feat_Y
	df['x1'] = tsne_results[:, 0]
	df['x2'] = tsne_results[:, 1]
	# df['x3'] = tsne_results[:, 2]
	df.to_csv('experiments/triplet_embed2D.csv')
	
	# df.to_csv('experiments/triplenet_embed3D.csv')
	# df = pd.read_csv('experiments/triplenet_embed2D.csv')
	
	df = pd.read_csv('experiments/triplet_embed2D.csv')

	plt.figure(figsize=(16,10))
	
	# ax = plt.axes(projection='3d')
	sns.scatterplot(
	    x="x1", y="x2",
	    data=df,
	    hue='label',
	    palette=sns.color_palette("hls", n_classes),
	    legend="full",
	    alpha=0.4
	)
	# ax.scatter3D(df['x1'], df['x2'], df['x3'], c=df['x3'], alpha=0.4)
	# plt.scatter(df['x1'], df['x2'], c=df['x2'], alpha=0.4)
	# min_x, max_x = np.min(feat_X), np.max(feat_X)
	# min_x, max_x = -10, 10

	# for c in range(len(np.unique(feat_Y))):
	# 	# ax.scatter(feat_X[feat_Y==c, 0], feat_X[feat_Y==c, 1], feat_X[feat_Y==c, 2], '.', alpha=0.5, c=colors[c], s=0.3)
	# 	plt.scatter(feat_X[feat_Y==c, 0], feat_X[feat_Y==c, 1], marker='.', alpha=0.5, c=colors[c], s=1.0)
	# plt.title('Triple Loss')

	# W = triplenet.cls_layer.get_weights()[0].T

	# x = np.linspace(min_x, max_x, 50)
	# y = W[0][1]*x + W[0][0]
	# plt.plot(x, y, c=colors[0])

	# x = np.linspace(min_x, max_x, 50)
	# y = W[1][1]*x + W[1][0]
	# plt.plot(x, y, c=colors[1])

	# x = np.linspace(min_x, max_x, 50)
	# y = W[2][1]*x + W[2][0]
	# plt.plot(x, y, c=colors[2])

	# x = np.linspace(min_x, max_x, 50)
	# y = W[3][1]*x + W[3][0]
	# plt.plot(x, y, c=colors[3])

	# x = np.linspace(min_x, max_x, 50)
	# y = W[4][1]*x + W[4][0]
	# plt.plot(x, y, c=colors[4])
	plt.savefig('experiments/embedding.png')
	# plt.show()
	# plt.clf()
	# plt.close()