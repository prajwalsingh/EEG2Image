import tensorflow as tf
import numpy as np
from tensorflow.keras import losses

# Add perceptual loss

def disc_hinge(dis_real, dis_fake):
	real_loss = -1.0 * tf.reduce_mean( tf.minimum(0.0, -1.0 + dis_real) )
	fake_loss = -1.0 * tf.reduce_mean( tf.minimum(0.0, -1.0 - dis_fake) )
	return (real_loss + fake_loss)/2.0

def gen_hinge(dis_fake):
	fake_loss = -1.0 * tf.reduce_mean( dis_fake )
	return fake_loss

def disc_loss(dis_real, dis_fake, dis_wrong=None):
	# real = tf.ones_like(dis_real)
	# fake = tf.zeros_like(dis_fake)
	real = tf.convert_to_tensor(np.random.randint(low=7, high=12, size=dis_real.shape)/10.0)
	fake = tf.convert_to_tensor(np.random.randint(low=0, high=3, size=dis_real.shape)/10.0)
	real_loss  = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(real, dis_real)
	fake_loss  = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(fake, dis_fake)
	#wrong_loss = losses.BinaryCrossentropy()(fake, dis_wrong)
	#total_loss = (real_loss + fake_loss + wrong_loss)/3.0
	# total_loss = tf.reduce_mean(real_loss**2 + fake_loss**2)
	total_loss = (real_loss + fake_loss)/2.0
	return total_loss

def gen_loss(dis_fake):
	real = tf.ones_like(dis_fake)
	fake_loss = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(real, dis_fake)
	return fake_loss

def critic_loss(D_real, D_fake):
	# real = -tf.ones_like(D_real)
	# fake =  tf.ones_like(D_fake)
	# return tf.reduce_mean(D_real*real) + tf.reduce_mean(D_fake*fake)
	return  (tf.reduce_mean(D_fake) - tf.reduce_mean(D_real))

# def gen_loss(D_fake, real_img, fake_img):
# 	# fake =  tf.ones_like(D_fake)
# 	# return tf.reduce_mean(D_fake*fake)
# 	# real_img = tf.clip_by_value(255.0*(real_img*0.5+0.5), 0.0, 255.0)
# 	# fake_img = tf.clip_by_value(255.0*(fake_img*0.5+0.5), 0.0, 255.0)
# 	return -tf.reduce_mean(D_fake) #+ 0.6 * tf.keras.losses.MeanSquaredError()(real_img, fake_img)

def wgan_gp_loss(D_real, D_fake, Y, Y_cap, model, batch_size):
	dloss = (tf.reduce_mean(D_fake) - tf.reduce_mean(D_real))
	lam   = 10
	eps   = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
	x_cap = eps * Y + (1-eps) * Y_cap
	with tf.GradientTape() as gptape:
		gptape.watch(x_cap)
		out = model.critic(x_cap, training=True)
	grad  = gptape.gradient(out, [x_cap])[0]
		# Fetching only x-gradient
	# grad_norm = tf.norm(grad, ord='euclidean', axis=1)
	# grad_pen  = tf.reduce_mean(tf.square(grad_norm - 1))
	grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
	grad_pen  = tf.reduce_mean((grad_norm - 1.0) ** 2)
	dloss     = dloss + lam * grad_pen
	return dloss
