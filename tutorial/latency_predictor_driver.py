import torch
import numpy as np
import time
import random
import os

from accuracy_predictor import AccuracyPredictor

from latency_table import LatencyTable
from evolution_finder import EvolutionFinder
import csv
import sys
sys.path.append("")
from ofa.model_zoo import ofa_net

sample_child_arch = {'wid': None, 'ks': [5, 5, 3, 3, 5, 3, 5, 7, 3, 3, 5, 7, 3, 5, 5, 7, 5, 7, 5, 5],
                          'e': [6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6], 'd': [4, 3, 3, 4, 4],
                          'r': [176]}
sample_latency = 37.044331933251044


# set random seed
random_seed = 10291284
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')


ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
print('The OFA Network is ready.')
data_loader = None

# Accuracy Predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

# Latency Predictor
target_hardware = 'gpu'
latency_table = LatencyTable(device=target_hardware, use_latency_table=False)
print('The Latency lookup table on %s is ready!' % target_hardware)


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
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
    'arch': 'compofa', ## change
}

# build the evolution finder
finder = EvolutionFinder(**params)

# import latency_predictor model
if '/latency_predictor' not in os.getcwd():
    os.chdir('./latency_predictor')
from latency_predictor import LatencyPredictor


def create_latency_dataset():
    # time to create dataset
    start = time.time()

    # create latency dataset
    number_of_datapoints = 10000
    latency_dataset = finder.create_latency_dataset(search_space='ofa', num_of_samples=number_of_datapoints)

    # create dataset csv file
    curr_hardware = 'Tesla_P40_GPU' #'Tesla_P40_GPU' #'note10_lookuptable' #'Intel_Xeon_CPU' #RTX_2080_Ti_GPU'
    filename = 'latency_predictor/datasets/' + curr_hardware + '_ofa.csv'
    with open(filename, 'w') as csv_file:
        w = csv.writer(csv_file)
        w.writerow(['child_arch', 'latency'])
        for i in range(len(latency_dataset['child_arch'])):
            child_arch = latency_dataset['child_arch'][i]
            latency = latency_dataset['latency'][i]
            #accuracy = latency_dataset['accuracy'][i]
            w.writerow([child_arch, latency])
    
    end = time.time()

    print('Wrote Latency Dataset to File: {}'.format(filename))
    print('Time to Create Dataset of {} points: {}'.format(number_of_datapoints, end-start))


def test_new_inference_time(model_path):
    RTX_checkpt = torch.load(model_path)
    RTX_model = LatencyPredictor().to(device)
    RTX_model.load_state_dict(RTX_checkpt)

    times = []
    for i in range(1000):
        start = time.time()
        prediction = RTX_model.predict_efficiency(sample_child_arch)
        end = time.time()

        model_time = end - start
        times.append(model_time)
    print('New Inference Time: {} seconds'.format(sum(times) / len(times)))


def test_old_inference_time():
    start = time.time()

    prediction = finder.efficiency_predictor.predict_efficiency(sample_child_arch)

    end = time.time()
    lookup_time = end - start
    print('Old Inference Time: {} seconds'.format(lookup_time))


if __name__ == '__main__':
    test_new_inference_time('../checkpoints/latency_prediction_model/checkpoint_note10_ofa.pt')#RTX_2080_Ti_GPU_ofa.pt')
    test_old_inference_time()
    # create_latency_dataset()
