import pdb
import os
import sys
import torch
import time
import math
import copy
import random
import argparse
import torch.nn as nn
import numpy as np

from torchvision import transforms, datasets
from matplotlib import pyplot as plt

sys.path.append("..")
from ofa.model_zoo import ofa_net
from ofa.utils import download_url
from accuracy_predictor import AccuracyPredictor
from flops_table import FLOPsTable
from latency_table import LatencyTable
from evolution_finder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized

parser = argparse.ArgumentParser()
# Possible values:
# 1. compofa
# 2. ofa_mbv3_d234_e346_k357_w1.0
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    help='OFA network')
"""
parser.add_argument(
    '-t',
    '--table',
    metavar='LATENCYTABLE',
    help='Flag to indicate the usage of latency table')
"""
args = parser.parse_args()
arch = {'compofa' : ('compofa', 'model_best_compofa_simple.pth.tar'),
        'compofa-elastic' : ('compofa-elastic', 'model_best_compofa_simple_elastic.pth.tar'),
        'ofa_mbv3_d234_e346_k357_w1.0' : ('ofa', 'ofa_mbv3_d234_e346_k357_w1.0'),
       }
MODEL_DIR = '/nethome/msahni7/shreya/compofa-NAS/ofa/checkpoints/%s' % (arch[args.net][1])

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '1' ## change
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

ofa_network = ofa_net(args.net, model_dir=MODEL_DIR, pretrained=True).cuda()
print('The OFA Network is ready.')

if cuda_available:
    imagenet_data_path = '/srv/data/datasets/ImageNet/' 

    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    if not os.path.isdir(imagenet_data_path):
        print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)

    print('The ImageNet dataset files are ready.')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

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


# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)
print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

# latency table
target_hardware = 'gpu'
latency_table = LatencyTable(device="gpu", use_latency_table=False, network=args.net) # change


""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 35  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
    'arch' : arch[args.net],
}

# build the evolution finder
finder = EvolutionFinder(**params)

result_lis = []
for latency in [15, 25, 35, 45]:
    st = time.time()
    finder.set_efficiency_constraint(latency)
    best_valids, best_info = finder.run_evolution_search()
    ed = time.time()
    result_lis.append(best_info)
print("NAS Completed!")
print(result_lis)

print(args.net)
# evaluate the searched model on ImageNet
if cuda_available:
    top1s = []
    latency_list = []
    for result in result_lis:
        _, net_config, latency = result
        print('Evaluating the sub-network with latency = %.1f ms on %s' % (latency, target_hardware))
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        top1s.append(top1)
        latency_list.append(latency)

print(top1s)
print("---------")
print(latency_list)
