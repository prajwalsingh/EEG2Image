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
import pickle
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment

style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'


# Thanks to: https://github.com/k-han/DTC/blob/master/utils/util.py
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

if __name__ == '__main__':

	n_channels  = 14
	n_feat      = 128
	batch_size  = 256
	test_batch_size  = 1
	n_classes   = 10

	# data_cls = natsorted(glob('data/thoughtviz_eeg_data/*'))
	# cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
	# idx2cls  = {value:key for key, value in cls2idx.items()}

	with open('data/eeg/image/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']


	# train_batch = load_complete_data('data/thoughtviz_eeg_data/*/train/*', batch_size=batch_size)
	# val_batch   = load_complete_data('data/thoughtviz_eeg_data/*/val/*', batch_size=batch_size)
	# test_batch  = load_complete_data('data/thoughtviz_eeg_data/*/test/*', batch_size=test_batch_size)
	train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
	# X, Y = next(iter(train_batch))
	# print(X.shape, Y.shape)
	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/best_ckpt', max_to_keep=5000)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
	START = int(triplenet_ckpt.step) // len(train_batch)
	if triplenet_ckptman.latest_checkpoint:
		print('Restored from the latest checkpoint, epoch: {}'.format(START))

	EPOCHS = 3000
	cfreq  = 178 # Checkpoint frequency
	test_loss = tf.keras.metrics.Mean()
	test_acc  = tf.keras.metrics.SparseCategoricalAccuracy()
	tq = tqdm(test_batch)
	feat_X = []
	feat_Y = []
	for idx, (X, Y) in enumerate(tq, start=1):
		_, feat = triplenet(X, training=False)
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
	df.to_csv('experiments/infer_triplet_embed2D.csv')	
	# df.to_csv('experiments/triplenet_embed3D.csv')
	# df = pd.read_csv('experiments/triplenet_embed2D.csv')
	
	df = pd.read_csv('experiments/infer_triplet_embed2D.csv')

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
	
	# plt.savefig('experiments/embedding.png')

	plt.show()

	# plt.clf()
	# plt.close()
	# featX = df[['x1', 'x2']].to_numpy()
	# print(featX.shape)

	kmeans = KMeans(n_clusters=n_classes,random_state=45)
	kmeans.fit(feat_X)
	labels = kmeans.labels_
	kmeanacc = cluster_acc(feat_Y, labels)
	# correct_labels = sum(feat_Y == labels)
	# print("Result: %d out of %d samples were correctly labeled." % (correct_labels, feat_Y.shape[0]))
	# kmeanacc = correct_labels/float(feat_Y.shape[0])
	print('Accuracy score: {0:0.2f}'. format(kmeanacc))