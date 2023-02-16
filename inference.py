import tensorflow as tf
# from utils import vis, load_batch#, load_data
from utils import load_complete_data, show_batch_images
from model import DCGAN, dist_train_step#, train_step
from tqdm import tqdm
import os
import shutil
import pickle
from glob import glob
from natsort import natsorted
import wandb
import numpy as np
import cv2

tf.random.set_seed(45)
np.random.seed(45)
# wandb.init(project='DCGAN_DiffAug_EDDisc_imagenet_128', entity="prajwal_15")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
clstoidx   = {}
idxtocls   = {}


# @tf.function
def get_code(path):
	path  = path.numpy().decode('utf-8')
	code  = np.zeros(shape=(max(clstoidx.values())+1,), dtype=np.float32)
	code[clstoidx[path.split(sep='/')[-2]]] = 1
	return tf.cast(code, dtype=tf.float32)


if __name__ == '__main__':

	# if len(glob('experiments/*'))==0:
	# 	os.makedirs('experiments/experiment_1/code/')
	# 	exp_num = 1
	# else:
	# 	exp_num = len(glob('experiments/*'))+1
	# 	os.makedirs('experiments/experiment_{}/code/'.format(exp_num))

	# exp_dir = 'experiments/experiment_{}'.format(exp_num)
	# for item in glob('*.py'):
	# 	shutil.copy(item, exp_dir+'/code')
	
	gpus = tf.config.list_physical_devices('GPU')
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0'], 
		cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	n_gpus = mirrored_strategy.num_replicas_in_sync
	# print(n_gpus)

	batch_size = 64
	latent_dim = 128
	input_res  = 64
	data_path  = 'data/images/ImageNet-Filtered/*/*'

	train_batch = load_complete_data(data_path, input_res=input_res, batch_size=batch_size)
	X, latent_Y = next(iter(train_batch))
	# print(latent_Y)
	latent_Y = latent_Y[:16]
	lr = 3e-4
	with mirrored_strategy.scope():
		model        = DCGAN()
		model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
		ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/best_ckpt', max_to_keep=30)
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	# print(ckpt.step.numpy())
	START         = int(ckpt.step.numpy()) // len(train_batch) + 1
	EPOCHS        = 1000#670#66
	model_freq    = 14#200#40
	t_visfreq     = 14#200#1500#40
	
	if ckpt_manager.latest_checkpoint:
		print('Restored from last checkpoint epoch: {0}'.format(START))

	for clidx in tqdm(range(10)):
		code = np.zeros(shape=(10,), dtype=np.float32)
		code[clidx] = 1
		code = np.expand_dims(code, axis=0)
		code = tf.cast(code, dtype=tf.float32)
		
		if not os.path.isdir('experiments/inference_result/{}'.format(clidx)):
			os.makedirs('experiments/inference_result/{}'.format(clidx))

		for _ in tqdm(range(256)):
			latent = tf.random.uniform(shape=(1, latent_dim), minval=-1, maxval=1)
			latent = tf.concat([latent, code], axis=-1)
			fake_img = mirrored_strategy.run(model.gen, args=(latent,))
			fake_img = fake_img[0].numpy()
			fake_img = np.uint8(np.clip(255*(fake_img * 0.5 + 0.5), 0.0, 255.0))
			fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
			cv2.imwrite('experiments/inference_result/{}/{}.png'.format(clidx, _), fake_img)