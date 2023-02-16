import torch
from torchmetrics.image.inception import InceptionScore
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception3
from torchvision.datasets import ImageFolder
import pathlib
from torch.utils.data import Dataset
from PIL import Image
from pytorch_gan_metrics import get_inception_score, get_fid

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
# if __name__ == '__main__':
# 	path = natsorted(glob('experiments/inception/294/*'))
# 	for sub_path in path:
# 		cls_name  = os.path.split(sub_path)[-1]
# 		imgs_path = natsorted(glob(sub_path+'/*'))
# 		inception = InceptionScore(splits=1)
# 		for img in tqdm(imgs_path):
# 			img = cv2.imread(img, 1)
# 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 			img = torch.permute(torch.Tensor(img).to(torch.uint8), (2, 0, 1))
# 			inception.update(torch.unsqueeze(img, axis=0))
# 			mean, std = inception.compute()
# 		print('Inception score for class {}: {}'.format(cls_name, mean, std))
# 		break
class MyDataset(Dataset):
	def __init__(self, image_paths, transform=None):
		self.image_paths = image_paths
		self.transform = transform

	def __getitem__(self, index):
		image_path = self.image_paths[index]
		x = Image.open(image_path)
		# y = self.get_class_label(image_path.split('/')[-1])
		if self.transform is not None:
			x = self.transform(x)
		# print(x.shape)
		return x

	def __len__(self):
		return len(self.image_paths)


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        img = img.to(torch.uint8)
        # print(img.dtype)
        # feature = inception(img)[0].view(img.shape[0], -1)
        # feature_list.append(feature.to('cpu'))
        inception.update(img)

    # features = torch.cat(feature_list, 0)

    return inception.compute()


if __name__ == '__main__':

	paths = natsorted(glob('experiments/inception/*'))

	for path in paths:
	    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	    # inception = InceptionScore(splits=10)
	    # inception = InceptionScore(feature=2048, splits=1)
	    # inception = inception.to(device)

	    transform = transforms.Compose(
	        [
	            # transforms.Resize( (299, 299) ),
	            # transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
	            transforms.ToTensor(),
	            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
	        ]
	    )

	    dset   = MyDataset(natsorted(glob(path+'/*')), transform)
	    loader = DataLoader(dset, batch_size=256, num_workers=8, shuffle=False)

	    # mean, std = extract_features(loader, inception, device)
	    mean, std = get_inception_score(loader)

	    print('Inception score for epoch {}: ({}, {})'.format(os.path.split(path)[-1], mean, std))

	    with open('experiments/inceptionscore_torch.txt', 'a') as file:
	    	file.write('Inception score for epoch {}: ({}, {})\n'.format(os.path.split(path)[-1], mean, std))