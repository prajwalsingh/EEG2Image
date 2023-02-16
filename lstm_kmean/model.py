import tensorflow as tf
from tensorflow.keras import Model, models, layers, regularizers
import tensorflow_addons as tfa

weight_decay = 1e-4

def enc_conv_block(filters, kernel, strides, padding, rate):
	return models.Sequential([
			layers.Conv1D(filters=filters, kernel_size=kernel, strides=strides, padding=padding),
			layers.Activation(activation='leaky_relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=rate)
		])


class TripleNet(Model):
	def __init__(self, n_classes=10, n_features=128):
		super(TripleNet, self).__init__()
		# filters   = [ 8,  16,  n_features]
		# ret_seq   = [ True, True, False]
		filters   = [ 32,  n_features]
		ret_seq   = [ True, False]
		# strides = [ 1,   2,  2,   2,]
		# kernel  = [ 7,   7,  3,   3,]
		# padding = ['same', 'same', 'same', 'same']
		self.enc_depth  = len(filters)
		# self.encoder    = [enc_conv_block(filters[idx], kernel[idx], strides[idx], padding[idx], rate=0.1) for idx in range(self.enc_depth)]
		self.encoder   = [layers.LSTM(units=filters[idx], return_sequences=ret_seq[idx]) for idx in range(self.enc_depth)]
		self.flat      = layers.Flatten()
		self.w_1       = layers.Dense(units=n_features, activation='leaky_relu')
		self.w_2       = layers.Dense(units=n_features)
		# self.feat_norm  = layers.BatchNormalization()
		# self.cls_layer  = layers.Dense(units=n_classes, kernel_regularizer=regularizers.l2(weight_decay))

	def call(self, x):
		for idx in range(self.enc_depth):
			x = self.encoder[idx]( x )
		# print(x.shape)
		x = feat = self.flat( x )
		# print(x.shape)
		# x = feat = self.feat_layer( x )
		# print(x.shape)
		# x = self.feat_norm( x )
		# x = self.cls_layer(x)
		# x = self.w_2( self.w_1( x ) )
		x = tf.nn.l2_normalize(x, axis=-1)

		return x, feat

@tf.function
def train_step(softnet, opt, X, Y):
	with tf.GradientTape() as tape:
		Y_emb, _ = softnet(X, training=True)
		# loss  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(Y, Y_emb)
		loss  = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
	variables = softnet.trainable_variables
	gradients = tape.gradient(loss, variables)
	opt.apply_gradients(zip(gradients, variables))
	return loss

@tf.function
def test_step(softnet, X, Y):
	Y_emb, _ = softnet(X, training=False)
	loss  = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
	# loss  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(Y, Y_emb)
	return loss