import os
import sys
sys.path.append("../ofa/")
import torch
import pickle
import random
import itertools
import numpy as np

from model_zoo import ofa_net
from elastic_nn.networks import OFAMobileNetV3
from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from imagenet_codebase.utils.pytorch_utils import get_net_info
from imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from imagenet_codebase.run_manager import ImagenetRunConfig, RunManager

GPU = 'all'
BATCH_SIZE = 192 
NUM_WORKERS = 20
ImagenetDataProvider.DEFAULT_PATH = '/netnvme/msahni7/ImageNet' 
MODEL_DIR = '/nethome/msahni7/manas/ofa-rescale/saved-runs/monotonic_1536/phase2/checkpoint/model_best.pth.tar' 
NUM_UNITS = 5

# Set GPU
if GPU == 'all':
    device_list = range(torch.cuda.device_count())
    GPU = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in GPU.split(',')]

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
BATCH_SIZE = BATCH_SIZE * max(len(device_list), 1)

# Load OFA
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
ofa_network = ofa_net("ofa")
run_config = ImagenetRunConfig(test_batch_size=BATCH_SIZE, n_worker=NUM_WORKERS)

def get_accuracy(subnet, res):
    run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
    run_config.data_provider.assign_active_img_size(res)
    run_manager.reset_running_statistics(net=subnet)
    loss, top1, top5 = run_manager.validate(net=subnet)
    return top1

def get_subnet(width_list, depth_list):
    kernel_stages = [3, 3, 5, 3, 3, 5]
    kernel_setting = []
    for k in kernel_stages[1:]:
        kernel_setting.extend([k]*4)
    ofa_network.set_active_subnet(e=width_list, d=depth_list, ks=kernel_setting)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    return subnet

def write_to_pickle(filename, population, accuracy):
    with open(filename, "wb") as f:
        pickle.dump(population, f)
        pickle.dump(accuracy, f)

def read_from_pickle(filename):
    with open(filename, "rb") as f:
        population = pickle.load(f)
        accuracy = pickle.load(f)
    return population, accuracy

