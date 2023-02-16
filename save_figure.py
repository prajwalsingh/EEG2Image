import cv2
import numpy as np
import os
import tensorflow as tf

def save_figure(X, save_path, categ=None):
	
	X = X.numpy()

	# if not os.path.isdir(save_path):
	# 	os.makedirs(os.path.join(save_path, 'train'))
	# 	os.makedirs(os.path.join(save_path, 'val'))
	# 	os.makedirs(os.path.join(save_path, 'test'))

	N      = X.shape[0]
	img_h  = X.shape[1]
	img_w  = X.shape[2]
	img_c  = X.shape[3]
	C      = 4
	R      = N // C
	h, w   = 0, 0
	canvas = np.ones((R*img_h, C*img_w, img_c), dtype=np.uint8)

	for img in X:
		img = np.uint8(np.clip(255*(img * 0.5 + 0.5), 0.0, 255.0))
		# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		canvas[h:h+img_h, w:w+img_w, :] = img
		w += img_w
		if w>=(C*img_w):
			w  = 0
			h += img_h

	cv2.imwrite(save_path, canvas)
	canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
	return canvas

def save_figure_condition(X, save_path, CN, lookup_dict, rev_lookup_dict, categ=None):
	
	X = X.numpy()

	# if not os.path.isdir(save_path):
	# 	os.makedirs(os.path.join(save_path, 'train'))
	# 	os.makedirs(os.path.join(save_path, 'val'))
	# 	os.makedirs(os.path.join(save_path, 'test'))

	N      = X.shape[0]
	img_h  = X.shape[1]
	img_w  = X.shape[2]
	img_c  = X.shape[3]
	C      = 4
	R      = N // C
	h, w   = 0, 0
	text_w = 12
	canvas = np.ones((R*img_h+R*text_w, C*img_w, img_c), dtype=np.uint8)

	for img, c in zip(X, CN):
		cimg = np.ones(shape=(img_h+text_w, img_w, img_c), dtype=np.uint8)*255
		c    = rev_lookup_dict[np.argmax(c.numpy())]
		img  = np.uint8(np.clip(255*(img * 0.5 + 0.5), 0.0, 255.0))
		img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cimg[text_w:, :, :] = img
		# cv2.putText(image, text, loc(x,y), font_type, font_scale, loc, thickness)
		cimg = cv2.putText(cimg, c, (img_w//2-20, 10), 1, 0.7, (0, 0, 0), 1)
		canvas[h:h+img_h+text_w, w:w+img_w, :] = cimg
		w += img_w
		if w>=(C*img_w):
			w  = 0
			h += img_h +text_w

	cv2.imwrite(save_path, canvas)
	canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
	return canvas

# import tensorflow as tf
# X = tf.convert_to_tensor((np.random.randint(0, 255, (7, 256, 256, 3)) - 127.5) / 127.5)
# save_results(X, 'result', 'train', 1)
