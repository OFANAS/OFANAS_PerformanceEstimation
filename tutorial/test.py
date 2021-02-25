import os
import sys
import pdb
sys.path.append("../ofa")
import math
import torch
import random
import numpy as np

from model_zoo import ofa_net
from torchvision import transforms, datasets
from imagenet_eval_helper import evaluate_ofa_subnet
from evolution_finder import ArchManager

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# set cuda
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

imagenet_data_path = '/netnvme/msahni7/ImageNet' 
ofa_network = ofa_net("ofa")

def get_data_loader():
	if cuda_available:
		# The following function build the data transforms for test
		def build_val_transform(size):
			return transforms.Compose([
				transforms.Resize(int(math.ceil(size / 0.875))),
				transforms.CenterCrop(size),
            	transforms.ToTensor(),
            	transforms.Normalize(
                	mean=[0.485, 0.456, 0.406],
                	std=[0.229, 0.224, 0.225]
            	),
        	])
			
		data_loader = torch.utils.data.DataLoader(
			datasets.ImageFolder(
				root=os.path.join(imagenet_data_path, 'val'),
				transform=build_val_transform(224)
				),
			batch_size=250,  # test batch size
        	shuffle=True,
        	num_workers=16,  # number of workers for the data loader
        	pin_memory=True,
        	drop_last=False,
    	)
		print('The ImageNet dataloader is ready.')
	else:
		data_loader = None
		print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')
	return data_loader

if __name__ == '__main__':
	data_loader = get_data_loader()
	am = ArchManager("compofa")
	net_config = am.random_sample()
	net_config = {'wid': None, 'ks': [3, 3, 5, 3, 7, 3, 3, 3, 5, 5, 3, 3, 5, 7, 7, 3, 3, 3, 7, 7], 'e': [3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [2, 3, 4, 3, 3], 'r': [192]} 
	top1 = evaluate_ofa_subnet(
		ofa_network,
		imagenet_data_path,
		net_config,
		data_loader,
		batch_size=250,
		device='cuda:0' if cuda_available else 'cpu')
	print(top1)

	
