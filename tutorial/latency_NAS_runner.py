import os
from collections import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy

from matplotlib import pyplot as plt

import sys
sys.path.append("..")
from ofa.model_zoo import ofa_net
from ofa.utils import download_url

from accuracy_predictor import AccuracyPredictor
from latency_table import LatencyTable
from evolution_finder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
# ofa_network_2 = ofa_net('ofa', pretrained=True)
print('The OFA Network is ready.')

if cuda_available:
    # path to the ImageNet dataset
    #print("Please input the path to the ImageNet dataset.\n")
    imagenet_data_path = '/home/msahni7/rhythm_arvind/imagenet_data/'

    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', model_dir='data')
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

if cuda_available:
    net_id = evaluate_ofa_specialized(imagenet_data_path, data_loader)
    print('Finished evaluating the pretrained sub-network: %s!' % net_id)
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

target_hardware = 'gpu'
latency_table = LatencyTable(device=target_hardware, use_latency_table=False)
print('The Latency lookup table on %s is ready!' % target_hardware)


# Load Latency Predictor Model
if '/latency_predictor' not in os.getcwd():
    os.chdir('./latency_predictor')
    sys.path.append(os.getcwd())
from latency_predictor import LatencyPredictor
latencypredictor_checkpt = torch.load('../checkpoints/latency_prediction_model/Finetune_Note_10/checkpoint_finetune_RTX_2080_Ti_GPU_ofa.pt')
#/Individual_trained_models/ofa/checkpoint_RTX_2080_Ti_GPU_ofa.pt
finetune=True
if finetune:
    key_names = ["model.0.weight", "model.0.bias", "model.2.weight", "model.2.bias", "model.4.weight", "model.4.bias", "model.6.weight", "model.6.bias"]
    new_state_dict = OrderedDict()
    i = 0
    for k, v in latencypredictor_checkpt.items():
        new_state_dict[key_names[i]] = v
        i = i + 1
    latencypredictor_checkpt = new_state_dict

latency_predictor = LatencyPredictor().cuda()
latency_predictor.load_state_dict(latencypredictor_checkpt)

option=1
efficiency_predictor = latency_predictor if option==1 else latency_table
if efficiency_predictor == latency_predictor:
        print('using latency prediction model')
elif efficiency_predictor == latency_table:
        print('using latency measurement')

""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 25  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': efficiency_predictor, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
    'arch' : 'compofa-elastic', ## change
}

# build the evolution finder
finder = EvolutionFinder(**params)

"""
# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))
"""

print('Starting NAS!')
result_lis = []
start = time.time()
for latency in [15, 20, 25, 30, 35]:
    finder.set_efficiency_constraint(latency)
    best_valids, best_info = finder.run_evolution_search()
    result_lis.append(best_info)
end = time.time()
print("Done!")

"""
# visualize the architecture of the searched sub-net
_, net_config, latency = best_info
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)
"""

print('Time: {}'.format(end-start))

# Evalute ImageNet Accuracy of chosen models
top1s_1 = [71.99200141906738, 73.83400161743164, 75.08400173187256, 75.56600158691407, 75.79200138092041, 76.46200187683105, 77.00800132751465, 77.09600151062011, 77.09200138092041, 77.27400165557862]
latency_list_1 = [14.915969134430984, 19.999238905682866, 24.74897687431021, 29.84660924139936, 34.780664078300624, 39.63810944460619, 44.780066593130584, 49.68762514010865, 54.796040548223594, 59.90540405076071]

# evaluate the searched model on ImageNet
if cuda_available:
    top1s_2 = []             ## change
    latency_list_2 = []      ## change
    for result in result_lis:
        _, net_config, latency = result
        print('Evaluating the sub-network with latency = %.1f ms on %s' % (latency, target_hardware))
        top1 = evaluate_ofa_subnet(
            ofa_network,                    ## change
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        top1s_2.append(top1)                ## change
        latency_list_2.append(latency)      ## change
    print('Accuracy')
    print(top1s_2)

    print('Latency')
    print(latency_list_2)
    """
    plt.figure(figsize=(4,4))
    plt.plot(latency_list_2, top1s_2, 'x-', marker='o', color='red',  linewidth=2, markersize=8, label='CompOFA_LatencyPred')
    plt.plot(latency_list_1, top1s_1, 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.plot([26, 45], [74.6, 76.7], '--', marker='+', linewidth=2, markersize=8, label='ProxylessNAS')
    plt.plot([15.3, 22, 31], [73.3, 75.2, 76.6], '--', marker='>', linewidth=2, markersize=8, label='MobileNetV3')
    plt.xlabel('%s Latency (ms)' % target_hardware, size=12)
    plt.ylabel('ImageNet Top-1 Accuracy (%)', size=12)
    plt.legend(['CompOFA_LatencyEsti', 'OFA', 'ProxylessNAS', 'MobileNetV3'], loc='lower right')
    plt.grid(True)
    plt.show()
    plt.savefig('NASexperiment_note10_LatencyEstimation.png')
    print('Successfully draw the tradeoff curve!')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')
"""


